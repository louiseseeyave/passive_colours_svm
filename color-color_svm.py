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
from flare.photom import m_to_flux


def get_data(color_set, hdf, colors, noise_std, replicate=1):

    # Define the colors data set
    data = np.zeros((ngal * replicate, len(colors[color_set])))

    # Loop until replicated required number of times
    i = 0
    while i < replicate:

        # Loop over colors
        for ii, (filt1, filt2) in enumerate(colors[color_set]):

            # Get magnitudes
            mag1 = hdf['galphotdust'][filt1][...]
            mag2 = hdf['galphotdust'][filt2][...]

            # Get flux
            flux1 = m_to_flux(mag1)
            flux2 = m_to_flux(mag2)

            # Add normally distributed noise
            if noise_std > 0:
                noise1 = np.random.normal(0, noise_std, size=mag1.shape)
                noise2 = np.random.normal(0, noise_std, size=mag2.shape)
                flux1 += noise1
                flux2 += noise2

            data[ngal * i: ngal * (i + 1), ii] = (flux1 / flux2)

    return data


def run_svm(data, truth, int_zs, label):

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

    print("High redshift galaxy? Accuracy: %.3f"
          % (metrics.accuracy_score(y_test, y_pred) * 100) + "%")

    cf_matrix = metrics.confusion_matrix(y_test, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',
                     cbar_kws={'label': '$N$'}, fmt="d")

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    ax.xaxis.set_ticklabels(['Contaminent', 'High-$z$'])
    ax.yaxis.set_ticklabels(['Contaminent', 'High-$z$'])

    plt.savefig("plots/highz_gal_classifier_%s.png" % label, bbox_inches="tight")

    plt.close()

    # ===================== Redshift binning =====================

    # Split into training and validation
    X_train, X_test, y_train, y_test = train_test_split(data[int_zs >= 5, :],
                                                        int_zs[int_zs >= 5],
                                                        test_size=0.3,
                                                        random_state=42)

    # Initialise the model
    clf = svm.SVC()

    # Train the model
    clf.fit(X_train, y_train)

    # Classify the prediction data set
    y_pred = clf.predict(X_test)

    print("Redshift bin accuracy: %.3f"
          % (metrics.accuracy_score(y_test, y_pred) * 100) + "%")

    cf_matrix = metrics.confusion_matrix(y_test, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',
                     cbar_kws={'label': '$N$'}, fmt="d")

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    ax.xaxis.set_ticklabels(np.unique(int_zs))
    ax.yaxis.set_ticklabels(np.unique(int_zs))

    plt.savefig("plots/redshift_bin_classifier_%s.png" % label, bbox_inches="tight")

    plt.close()

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

path = "Euclid.h5"

hdf = h5py.File(path, "r")

print("Filters:", list(hdf['galphotdust'].keys()))

ngal = hdf['galphotdust']['z'].size
print("There are", ngal, '"galaxies"')

# Get redshifts
zs = hdf['galphotdust']['z'][...]
int_zs = np.int32(zs)
print("Redshift truth", np.unique(int_zs, return_counts=True))

# Define the truth array (z > 5)
truth = np.zeros(ngal)
truth[zs >= 5] = 1

print("Galaxy truth", np.unique(truth, return_counts=True))

# Define band wavelengths
wls = {'Euclid_H': (2 - (2 - 1.544)) * 1000,
       'Euclid_J': (1.544 - (1.544 - 1.192)) * 1000,
       'Euclid_VIS': (0.9 - (0.9 - 0.55)) * 1000,
       'Euclid_Y': (1.192 - (1.192 - 0.95)) * 1000,
       'LSST_g': 551.9 - (551.9 - 401.5), 'LSST_i': 818.0 - (818.0 - 691.0),
       'LSST_r': 691.0 - (691.0 - 552.0), 'LSST_u': 393.5 - (393.5 - 320.5),
       'LSST_y': 1084.5 - (1084.5 - 923.8), 'LSST_z': 923.5 - (923.5 - 818.0)}

# Now define the colors
colors = {0: (('Euclid_VIS', 'LSST_z'), ('LSST_z', 'Euclid_Y'),
              ('Euclid_Y', 'Euclid_J'), ('Euclid_J', 'Euclid_H'),
              ('Euclid_H', 'LSST_u')),
          1: (('Euclid_VIS', 'LSST_z'), ('LSST_z', 'Euclid_Y'),
              ('Euclid_Y', 'Euclid_J'), ('Euclid_J', 'Euclid_H'))}

# Which color set are we running with?
color_set = int(sys.argv[1])
noise = [0, 0.05, 0.1, 0.5, 1][int(sys.argv[2])]
replicate = int(sys.argv[2])

# Define the colors data set
data = get_data(color_set, hdf, colors, noise_std=noise, replicate=replicate)

hdf.close()

run_svm(data, truth, int_zs, 
        "Euclid_LSST_colorset-%d_noise-%.1f_replicate-%d" % (color_set,
                                                             noise,
                                                             replicate))
