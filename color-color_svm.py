import sys
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
# import umap

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
from flare.photom import m_to_flux, flux_to_m


def get_data(color_set, path, colors, noise_std, replicate=1, replicate_z=0):

    # Open hdf5 file
    hdf = h5py.File(path, "r")

    print("Filters:", list(hdf['galphotdust'].keys()))

    ngal = hdf['galphotdust']['z'].size
    print("There are", ngal, '"galaxies"')

    # Get redshifts
    zs = hdf['galphotdust']['z'][...]

    # Initialise class arrays
    int_zs = np.zeros(ngal + zs[zs >= replicate_z].size * replicate)
    truth = np.zeros(ngal + zs[zs >= replicate_z].size * replicate)

    # Define the colors data set
    data = np.zeros((ngal + zs[zs >= replicate_z].size * replicate,
                     len(colors[color_set])))

    # Loop until replicated required number of times
    i = 0
    current_ngal = 0
    while i < replicate:

        if i > 0:
            okinds = zs >= replicate_z
        else:
            okinds = np.ones(ngal, dtype=bool)

        this_ngal = zs[okinds].size

        # Loop over colors
        for ii, (filt1, filt2) in enumerate(colors[color_set]):

            # Get magnitudes
            mag1 = hdf['galphotdust'][filt1][okinds]
            mag2 = hdf['galphotdust'][filt2][okinds]

            # Get flux
            flux1 = m_to_flux(mag1)
            flux2 = m_to_flux(mag2)

            # Add normally distributed noise
            if noise_std > 0:
                noise1 = np.random.normal(0, noise_std, size=mag1.shape)
                noise2 = np.random.normal(0, noise_std, size=mag2.shape)
                flux1 += noise1
                flux2 += noise2

            # And back to magnitude
            new_mag1 = flux_to_m(flux1)
            new_mag2 = flux_to_m(flux2)

            # Now calculate the color
            data[current_ngal: current_ngal + this_ngal, ii] = (new_mag1
                                                                - new_mag2)

        # Compute integer redshift bin
        int_zs[current_ngal: current_ngal + this_ngal] = np.int32(zs[okinds])

        i += 1
        current_ngal += this_ngal

    hdf.close()

    # Define the truth array (z >= 5)
    truth[int_zs >= 5] = 1

    # Remove any galaxies that were made into nans
    nan_okinds = np.ones(data.shape[0], dtype=bool)
    for i in range(len(colors[color_set])):
        nan_okinds = np.logical_and(nan_okinds, ~np.isnan(data[:, i]))
    data = data[nan_okinds, :]
    truth = truth[nan_okinds]
    int_zs = int_zs[nan_okinds]

    print("Redshift truth", np.unique(int_zs, return_counts=True))
    print("Galaxy truth", np.unique(truth, return_counts=True))

    print("There are", truth.size, '"galaxies" after replication')

    return data, truth, int_zs


def run_svm(data, truth, int_zs, label, colors):

    print("Got data with shapes", data.shape, truth.shape, int_zs.shape)

    # ===================== Is high redshift galaxy? =====================

    # Split into training and validation
    X_train, X_test, y_train, y_test = train_test_split(data, truth,
                                                        test_size=0.3,
                                                        random_state=42)

    # Initialise the model
    clf = svm.SVC()

    # Train the model
    clf.fit(X_train, y_train)

    # Classify the prediction data set
    y_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("High redshift galaxy? Accuracy: %.3f"
          % acc + "%")

    cf_matrix = metrics.confusion_matrix(y_test, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',
                     cbar_kws={'label': '$N$'}, fmt="d")

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    ax.xaxis.set_ticklabels(['Contaminent', 'High-$z$'])
    ax.yaxis.set_ticklabels(['Contaminent', 'High-$z$'])

    ax.text(0.95, 0.95, '%.2f' % acc + "%",
                 bbox=dict(boxstyle="round,pad=0.3", fc='w',
                           ec="k", lw=1, alpha=0.8),
                 transform=ax.transAxes,
                 horizontalalignment='right',
                 fontsize=8)


    plt.savefig("plots/highz_gal_classifier_%s.png" % label, bbox_inches="tight")

    plt.close()

    names = [i + " - " + j for i, j in colors]

    f_importances(clf.coef_, names, "highz", label=label)

    # ===================== Redshift binning =====================

    # Remove low redshift data to emulate a hierarchical approach with
    # perfect high redshift classification
    okinds = int_zs >= 5
    highz_data = data[okinds, :]
    highz_int_zs = int_zs[okinds]

    # Split into training and validation
    X_train, X_test, y_train, y_test = train_test_split(highz_data,
                                                        highz_int_zs,
                                                        test_size=0.3,
                                                        random_state=42)

    # Initialise the model
    clf = svm.SVC()

    # Train the model
    clf.fit(X_train, y_train)

    # Classify the prediction data set
    y_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("Redshift bin accuracy: %.3f"
          % acc + "%")

    cf_matrix = metrics.confusion_matrix(y_test, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',
                     cbar_kws={'label': '$N$'}, fmt="d")

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    ax.text(0.95, 0.95, '%.2f' % acc + "%",
                 bbox=dict(boxstyle="round,pad=0.3", fc='w',
                           ec="k", lw=1, alpha=0.8),
                 transform=ax.transAxes,
                 horizontalalignment='right',
                 fontsize=8)

    ax.xaxis.set_ticklabels(np.unique(highz_int_zs))
    ax.yaxis.set_ticklabels(np.unique(highz_int_zs))

    plt.savefig("plots/redshift_bin_classifier_%s.png" % label, bbox_inches="tight")

    plt.close()

    f_importances(clf.coef_, names, "redshift", label=label)

    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(data)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.grid(True)
    #
    # ax.scatter(embedding[:, 0], embedding[:, 1],
    #            c=truth, cmap="coolwarm")
    #
    # ax.set_aspect('equal', 'datalim')
    #
    # plt.savefig("highz_gal_umap_%s.png" % label, bbox_inches="tight")
    #
    # plt.close()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.grid(True)
    #
    # im = ax.scatter(embedding[:, 0], embedding[:, 1],
    #                 c=zs, cmap="plasma")
    #
    # ax.set_aspect('equal', 'datalim')
    #
    # cbar = fig.colorbar(im)
    # cbar.set_label("$z$")
    #
    # plt.savefig("redshift_umap_%s.png" % label, bbox_inches="tight")
    #
    # plt.close()


def f_importances(coef, names, class_type, label):
    """ https://stackoverflow.com/questions/41592661/
        determining-the-most-contributing-features-for-svm
        -classifier-in-sklearn """

    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.barh(range(len(imp)), imp[::-1], align='center')
    ax.set_yticks(range(len(imp)), names[::-1])

    fig.savefig("plots/feature_importance_type%s_%s.png" % (class_type, label),
                bbox_inches="tight")

    plt.close(fig)

# ===================== Euclid Catalogue =====================

# # Define constants
# cat_str = "MCMC_cat.csv"
#
# # Open dataframe
# df = pd.read_csv(cat_str)
#
# # Get filters, truth and redshift
# filts = [i for i in list(df)[1:] if i[-3:] == "mag"]#[0:6]
# truth = df["Truth"].to_numpy()
# zs = df["z"].to_numpy()
# int_zs = np.int32(df["z"].to_numpy())
#
# print("Truth", np.unique(truth, return_counts=True))
# print("Redshift bins", np.unique(int_zs, return_counts=True))
#
# # Define the colors
# colors = (('VIS mag', 'zHSC mag'), ('zHSC mag', 'Y mag'),
#           ('Y mag', 'J mag'), ('J mag', 'H mag'), ('H mag', 'irac1 mag'))
# # colors = []
# # for i, f1 in enumerate(filts[:6]):
# #     for f2 in filts[i + 1:6]:
# #         colors.append((f1, f2))
# # for i, f1 in enumerate(filts[6:-1]):
# #     for f2 in filts[6 + i + 1:]:
# #         colors.append((f1, f2))
#
# ngal = len(df)
#
# # Print some nice things
# print("Filters:", filts)
# print("Colors:", colors)
# print("With", len(df), '"galaxies"')
#
# # Define the colors data set
# data = np.zeros((len(df), len(colors)))
# for i, (filt1, filt2) in enumerate(colors):
#     data[:, i] = df[filt1] - df[filt2]
#
# run_svm(data, truth, int_zs, "Euclid_Flag")

# ===================== Aaron's SAM =====================

# Define filepath
path = "Euclid.h5"

# Define band wavelengths
wls = {'Euclid_H': (2 - (2 - 1.544)) * 1000,
       'Euclid_J': (1.544 - (1.544 - 1.192)) * 1000,
       'Euclid_VIS': (0.9 - (0.9 - 0.55)) * 1000,
       'Euclid_Y': (1.192 - (1.192 - 0.95)) * 1000,
       'LSST_g': 551.9 - (551.9 - 401.5), 'LSST_i': 818.0 - (818.0 - 691.0),
       'LSST_r': 691.0 - (691.0 - 552.0), 'LSST_u': 393.5 - (393.5 - 320.5),
       'LSST_y': 1084.5 - (1084.5 - 923.8), 'LSST_z': 923.5 - (923.5 - 818.0)}

# Create list of wavelengths
ls = []
fs = []
for key in wls:
    ls.append(wls[key])
    fs.append(key)

# Order filters
ls = np.array(ls)
fs = np.array(fs)
sinds = np.argsort(ls)
fs = fs[sinds]

# Get every possible colour (ifilt - each redder ifilt)
cs = []
for i, f1 in enumerate(fs[:-1]):
    for j, f2 in enumerate(fs[i + 1:]):
        cs.append((f1, f2))

# Now define the colors
colors = {0: (('Euclid_VIS', 'LSST_z'), ('LSST_z', 'Euclid_Y'),
              ('Euclid_Y', 'Euclid_J'), ('Euclid_J', 'Euclid_H'),
              ('Euclid_H', 'LSST_u')),
          1: (('Euclid_VIS', 'LSST_z'), ('LSST_z', 'Euclid_Y'),
              ('Euclid_Y', 'Euclid_J'), ('Euclid_J', 'Euclid_H')),
          2: cs}

# Which color set are we running with?
color_set = int(sys.argv[1])
noise = [0, 0.5, 1., 5., 10., 25., 50., 100.][int(sys.argv[2])]  # in nJy
replicate = int(sys.argv[3])
replicate_z = int(sys.argv[4])  # replicate only galaxies above this z

if noise == 0 and replicate > 1:
    print("Pointless to regenerate the same points without noise")
    replicate = 1

# Define the colors data set
data, truth, int_zs = get_data(color_set, path, colors,
                               noise_std=noise, replicate=replicate,
                               replicate_z=replicate_z)

# Run SVM
run_svm(data, truth, int_zs,
        "Euclid_LSST_nD%d_colorset-%d_"
        "noise-%.1f_replicate-%d" % (len(colors[color_set]), color_set,
                                     noise, replicate), colors[color_set])
