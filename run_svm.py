import numpy as np
from color_color_svm_no_noise_train import get_data, get_mask, run_svm

# --------------------------------------------------------------------


# filters
miri = ['F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W',
        'F2550W', 'F560W', 'F770W']
nircam = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F182M',
          'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F356W', 'F360M',
          'F410M', 'F430M', 'F444W', 'F460M', 'F480M']
nircam_lw = ['F250M', 'F277W', 'F300M', 'F356W', 'F360M', 'F410M',
             'F430M', 'F444W', 'F460M', 'F480M']
all_filters = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M',
               'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M',
               'F356W', 'F360M', 'F410M', 'F430M', 'F444W', 'F460M',
               'F480M', 'F560W', 'F770W','F1000W', 'F1130W', 'F1280W',
               'F1500W', 'F1800W', 'F2100W', 'F2550W']

# minimum detectable flux for each filter [nJy]
limits = [22.5, 15.3, 13.2, 19.4, 10.6, 21.4, 16.1, 9.1, 14.9, 32.1,
          14.3, 25.8, 12.1, 20.7, 24.7, 50.9, 23.6, 46.5, 67.9, 130,
          240, 520, 1220, 920, 1450, 2970, 5140, 17300]

flux_threshold = dict(zip(all_filters, limits))


# --------------------------------------------------------------------


# get dictionary of color sets
color_dict = {}

# all nircam filters
set = []
for i, f1 in enumerate(nircam[:-1]):
    for j, f2 in enumerate(nircam[i + 1:]):
        set.append((f1, f2))
print(set)
color_dict['nircam'] = set

# nircam long wavelength filters + miri F560W and F770W
set = []
nircamlw_f560w_f770w = nircam_lw + ['F560W', 'F770W']
for i, f1 in enumerate(nircamlw_f560w_f770w[:-1]):
    for j, f2 in enumerate(nircamlw_f560w_f770w[i + 1:]):
        set.append((f1, f2))
print(set)
color_dict['nircamlw_f560w_f770w'] = set


# --------------------------------------------------------------------


# RUN SVM


path = '/cosma7/data/dp004/dc-seey1/data/flares/myversion/passive_data.hdf5'


# parameters *********************
# specify filters to use
#color_set = 'nircamlw_f560w_f770w' # choose color set
#keep = get_mask(nircamlw_f560w_f770w, flux_threshold, path) # get mask
color_set = 'nircam'
keep = get_mask(nircam, flux_threshold, path) # get mask

# specify class weight (None, 'balanced', dictionary)
class_weight = 'balanced'

# specify kernel (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)
kernel = 'linear'

# specify label
#label = color_set
label = '_'.join((color_set, kernel, class_weight))
# ********************************


data, truth = get_data(color_dict, color_set, path, keep)

noise = [0.]  # [nJy] (no noise)

feature_names = np.array(color_dict[color_set])
feature_names = [f'{x}-{y}' for (x,y) in feature_names]

run_svm(data, truth, label, noise, feature_names, kernel=kernel,
        class_weight=class_weight)
