import sys
import os

import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
# import umap

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
from flare.photom import m_to_flux, flux_to_m


def get_data(colors, color_set, path, keep):

    """
    colors: dictionary of color sets
    color_set: integer of chosen color set
    path: location of flux data, e.g. 
    /cosma7/data/dp004/dc-seey1/data/flares/myversion/passive_data.hdf5
    keep: mask for removing low mass and low brightness galaxies
    """
    
    # Open hdf5 file
    hdf = h5py.File(path, "r")

    ngal = hdf['combined/ssfr'].size
    print("There are", ngal, "galaxies before mass and flux cuts")

    # Get ssfr
    ssfr = np.array(hdf['combined/ssfr'])[keep]

    # Initialise class arrays
    truth = np.zeros(ngal)[keep]

    print("There are", len(truth), "galaxies after mass and flux cuts")

    # Define the colors data set
    data = np.zeros((len(truth), len(colors[color_set])))

    # Loop over colors
    for ii, (filt1, filt2) in enumerate(colors[color_set]):

        # Get fluxes
        flux1 = np.array(hdf[f'combined/{filt1}'])
        flux2 = np.array(hdf[f'combined/{filt2}'])

        # Now calculate the color
        data[:, ii] =  2.5*np.log10(flux2/flux1)[keep]

    hdf.close()

    # Define the truth array (ssfr < -1)
    truth[ssfr < -1] = 1

    print("No. of passive galaxies (truth):", np.sum(truth))

    return data, truth


def get_mask(filter_list, flux_threshold, path):

    """implement mass cut and min flux for each filter"""

    fp = h5py.File(path, 'r') # open file
    
    mstar = np.log10(fp['combined/mstar30']) + 10
    keep = mstar > 9 # mass cut

    for filt in filter_list:
        values = np.array(fp[f'combined/{filt}'])
        keep = np.logical_and(keep, values > flux_threshold[filt])

    return keep


def random_sample(data, truth, oversample=0.0, undersample=0.0,
                  shrinkage=None):

    """
    randomly over or undersample *exact* data points
    please note that oversampling takes place first
    oversample: desired fraction of  minority / majority samples
    undersample: desired fraction of  minority / majority samples
    shrinkage: dispersion factor for oversampling (smoothed bootstrap)
    """

    print(f'Before: passive frac {np.sum(truth)/len(truth)}')
    
    if (undersample < oversample) & (undersample > 0.0):
        print('Error: oversample > undersample')
        return

    if oversample > 0.0:
        over = RandomOverSampler(sampling_strategy=oversample,
                                 shrinkage=shrinkage)
        data, truth = over.fit_resample(data, truth)

    if undersample > 0.0:
        under = RandomUnderSampler(sampling_strategy=undersample)
        data, truth = under.fit_resample(data, truth)

    print(f'After: passive frac {np.sum(truth)/len(truth)}')
    print(f'{len(truth)} galaxies, {np.sum(truth)} passive')

    return data, truth


def run_svm(data, truth, label, noise, feature_names, kernel='linear',
            class_weight=None, test_size=0.3):

    print("Got data with shapes", data.shape, truth.shape)

    # ================= Is the galaxy passive? =======================

    # Split into training and validation
    X_train, X_test, y_train, y_test = train_test_split(data, truth,
                                                        test_size=test_size,
                                                        random_state=42)

    # Initialise the model
    clf = svm.SVC(kernel=kernel, class_weight=class_weight)

    # Train the model
    clf.fit(X_train, y_train)

    # Loop over noise levels for prediction
    for n in noise:

        if n > 0:
            noise_arr = np.random.normal(0, n, X_test.shape)
            Xf_test = m_to_flux(X_test)
            Xf_test += noise_arr
            X_test = flux_to_m(Xf_test)

        # Classify the prediction data set
        y_pred = clf.predict(X_test)

        acc = metrics.accuracy_score(y_test, y_pred) * 100
        prec = metrics.precision_score(y_test, y_pred) * 100
        recall = metrics.recall_score(y_test, y_pred) * 100
        print(f"Ran SVM with {n:.1f} noise std \
        Accuracy: {acc:.3f}% Precision: {prec:.3f}% Recall: {recall:.3f}%")

        # Save scores in a dictionary
        with open('scores/passive_classifier_scores.pkl', 'rb') as f:
            scores_dict = pickle.load(f)
        scores_dict[label] = {'accuracy': acc, 'precision': prec,
                              'recall': recall}
        with open('scores/passive_classifier_scores.pkl', 'wb') as new_f:
            pickle.dump(scores_dict, new_f)

        cf_matrix = metrics.confusion_matrix(y_test, y_pred)

        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',
                         cbar_kws={'label': '$N$'}, fmt="d")

        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Actual Class')

        ax.xaxis.set_ticklabels(['Contaminent', 'Passive'])
        ax.yaxis.set_ticklabels(['Contaminent', 'Passive'])

        ax.text(0.95, 0.95, '%.2f' % acc + "%",
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k",
                          lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        plt.savefig("figures/random_sample/passive_gal_classifier_%s_noise-%.1f.png"
                    % (label, n), bbox_inches="tight", dpi=300)

        plt.close()

    # =============== Plot feature importance ========================

    if kernel!='linear': return
    
    imp = np.array(clf.coef_[0])
    names = feature_names
    imp, names = zip(*sorted(zip(imp,names)))

    fig = plt.figure(figsize=(len(names)/6, len(names)/3))
    ax = fig.add_subplot()

    ax.barh(range(len(imp)), imp, align='center')
    ax.set(yticks=range(len(imp)), yticklabels=names)

    fig.savefig("figures/feature_importance_%s.png" % (label),
                 bbox_inches="tight", dpi=300)

    plt.close(fig)

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



# EXAMPLE CODE -------------------------------------------------------


# Define filepath
#path = "Euclid.h5"

# Define band wavelengths
#wls = {'Euclid_H': (2 - (2 - 1.544)) * 1000,
#       'Euclid_J': (1.544 - (1.544 - 1.192)) * 1000,
#       'Euclid_VIS': (0.9 - (0.9 - 0.55)) * 1000,
#       'Euclid_Y': (1.192 - (1.192 - 0.95)) * 1000,
#       'LSST_g': 551.9 - (551.9 - 401.5), 'LSST_i': 818.0 - (818.0 - 691.0),
#       'LSST_r': 691.0 - (691.0 - 552.0), 'LSST_u': 393.5 - (393.5 - 320.5),
#       'LSST_y': 1084.5 - (1084.5 - 923.8), 'LSST_z': 923.5 - (923.5 - 818.0)}

# Create list of wavelengths
#ls = []
#fs = []
#for key in wls:
#    ls.append(wls[key])
#    fs.append(key)

# Order filters
#ls = np.array(ls)
#fs = np.array(fs)
#sinds = np.argsort(ls)
#fs = fs[sinds]

# Get every possible colour (ifilt - each redder ifilt)
#cs = []
#for i, f1 in enumerate(fs[:-1]):
#    for j, f2 in enumerate(fs[i + 1:]):
#        cs.append((f1, f2))

# Now define the colors
#colors = {0: (('Euclid_VIS', 'LSST_z'), ('LSST_z', 'Euclid_Y'),
#              ('Euclid_Y', 'Euclid_J'), ('Euclid_J', 'Euclid_H'),
#              ('Euclid_H', 'LSST_u')),
#          1: (('Euclid_VIS', 'LSST_z'), ('LSST_z', 'Euclid_Y'),
#              ('Euclid_Y', 'Euclid_J'), ('Euclid_J', 'Euclid_H')),
#          2: cs}

# Which color set are we running with?
#color_set = 0  # int(sys.argv[1])
#noise = [0.0, 1, 5, 10, 25., 50., 100.]  # in nJy

# Define the colors data set
#data, truth, int_zs = get_data(color_set, path, colors)

# Run SVM
#run_svm(data, truth, int_zs,
#        "Euclid_LSST_nD%d_colorset-%d_" % (len(colors[color_set]), color_set),
#        colors[color_set],  noise)
