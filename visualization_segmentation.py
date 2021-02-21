import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes
from opt_parameters import clf, min_plant_size
plt.rcParams.update({'font.size': 14})

def read_data(dates=None, path_to_data="data//", x_start=None, y_start=None, crop_size=None):

    bands_test = dict()
    ground_truth_test = dict()
    mask_test = dict()

    for date in dates:

        bands_test[date] = dict()

        for channel in ["Red", "Green", "Blue"]:
            bands_test[date][channel] = np.load(path_to_data + date + '_test_' + channel + '.npy')
            bands_test[date][channel] = np.where(bands_test[date][channel] > 255, 255, bands_test[date][channel])

        bands_test[date]["RGB"] = np.dstack(
            (bands_test[date]["Red"], bands_test[date]["Green"], bands_test[date]["Blue"])).astype('uint8')

        ground_truth_test[date] = np.load(path_to_data + date + '_test_GT.npy')

        mask_test[date] = np.load(path_to_data + date + '_test_mask.npy')

        x_end = x_start + crop_size
        y_end = y_start + crop_size

    for date in dates:
        ground_truth_test[date] = ground_truth_test[date][x_start:x_end, y_start:y_end]
        for channel in ["Red", "Green", "Blue", "RGB"]:
            bands_test[date][channel] = bands_test[date][channel][x_start:x_end, y_start:y_end]

    return bands_test, ground_truth_test

dates = ['2019_07_25', '2019_09_20', '2019_10_11']
cmap = ListedColormap(["lawngreen", "forestgreen", "red", "darkorange"])
clf_types = ['ExG', 'Linear', 'VDVI', 'Fraction', 'Mask_RCNN']
x_start = 30
y_start = 2020
crop_size = 300
x_end = x_start + crop_size
y_end = y_start + crop_size
path_to_data = "data//Segmentation//"

bands_test, ground_truth_test = read_data(
    dates=dates, path_to_data=path_to_data, x_start=x_start, y_start=y_start, crop_size=crop_size)

output_images = np.zeros(shape=(3, 6)).tolist()
for counter, date in enumerate(dates):
    output_images[counter][0] = bands_test[date]["RGB"]

for counter_date, date in enumerate(dates):
    GT = ground_truth_test[date]
    for counter_clf, clf_type in enumerate(clf_types):
        R = bands_test[date]["Red"]
        G = bands_test[date]["Green"]
        B = bands_test[date]["Blue"]

        if clf_type == "ExG":
            mask_predicted = 2 * G - R - B
            mask_predicted = np.nan_to_num(mask_predicted)
            mask_predicted = (mask_predicted > clf[date][clf_type]['threshold'])

        if clf_type == "VDVI":
            mask_predicted = (2 * G - R - B) / (2 * G + R + B)
            mask_predicted = (mask_predicted > clf[date][clf_type]['threshold'])

        if clf_type == "Linear":
            coefs = clf[date][clf_type]['coefs']
            mask_predicted = coefs[0] * R + coefs[1] * G + coefs[2] * B + coefs[3]
            mask_predicted = (mask_predicted > clf[date][clf_type]['threshold'])

        if clf_type == "Fraction":
            coefs = clf[date][clf_type]['coefs']
            mask_predicted = (coefs[0] * R + coefs[1] * G + coefs[2] * B) / (coefs[3] * R + coefs[4] * G + coefs[5] * B)
            mask_predicted = (mask_predicted > 0.2)

        if clf_type == "Mask_RCNN":
            mask_predicted = np.load(path_to_data + date.replace("_", "-") + "_Mask_RCNN_output.npy")[x_start:x_end, y_start:y_end]

        mask_predicted = remove_small_objects(mask_predicted, min_size=min_plant_size[date])
        mask_predicted = remove_small_holes(mask_predicted, area_threshold=min_plant_size[date])

        output = np.zeros(shape=mask_predicted.shape)
        output += 1 * np.multiply(mask_predicted == 1, GT == 1)
        output += 2 * np.multiply(mask_predicted == 0, GT == 0)
        output += 3 * np.multiply(mask_predicted == 1, GT == 0)  # red - false positive
        output += 4 * np.multiply(mask_predicted == 0, GT == 1)  # orange = false negative

        output_images[counter_date][counter_clf + 1] = output

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(16, 8), subplot_kw={'xticks': [], 'yticks': []})
for row in range(3):
    for col in range(6):
        print(output_images[row][col].shape)
        axs[row, col].imshow(output_images[row][col], cmap=cmap)

        if col == 0:
            axs[row, col].set(ylabel=dates[row])

        list_clf_names = ["RGB"] + clf_types
        list_clf_names[2] = "Optimized Linear"
        list_clf_names[4] = "Optimized Fraction"

        if row == 0:
            axs[row, col].set_title(list_clf_names[col])

plt.tight_layout()
# plt.show()
plt.savefig("Diff_indicators_comparision.png", dpi=300)
