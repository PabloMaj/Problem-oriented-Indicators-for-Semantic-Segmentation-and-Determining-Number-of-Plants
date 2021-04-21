import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes
from opt_parameters import clf, min_plant_size
plt.rcParams.update({'font.size': 18})

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

    return bands_test, ground_truth_test, mask_test

dates = ['2019_07_25', '2019_09_20', '2019_10_11']
clf_types = ['ExG', 'Linear', 'VDVI', 'Fraction', 'Mask_RCNN']
path_to_data = "data//Segmentation//"

bands_test, ground_truth_test, mask_test = read_data(dates=dates, path_to_data=path_to_data)

VARI_mean_results = open("VARI_mean_results.txt", "w")

for counter_date, date in enumerate(dates):
    GT = ground_truth_test[date]
    for counter_clf, clf_type in enumerate(clf_types):

        R = bands_test[date]["Red"]
        G = bands_test[date]["Green"]
        B = bands_test[date]["Blue"]
        VARI = (G - R) / (G + R - B)
        VARI_mean_true = np.ma.array(VARI, mask=np.logical_not(ground_truth_test[date])).mean()

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
            mask_predicted = np.load(path_to_data + date.replace("_", "-") + "_Mask_RCNN_output.npy")

        mask_predicted = remove_small_objects(mask_predicted, min_size=min_plant_size[date])
        mask_predicted = remove_small_holes(mask_predicted, area_threshold=min_plant_size[date])

        VARI_mean_predict = np.ma.array(VARI, mask=np.logical_not(mask_predicted)).mean()
        VARI_error = np.abs(VARI_mean_predict - VARI_mean_true) * 100 / VARI_mean_true
        print(f"{date}\t{clf_type}\t{np.round(VARI_mean_predict, 4)}\t{np.round(VARI_mean_true, 4)}\t{np.round(VARI_error, 2)}\n")
        VARI_mean_results.write(f"{date}\t{clf_type}\t{np.round(VARI_mean_predict, 4)}\t{np.round(VARI_mean_true, 4)}\t{np.round(VARI_error, 2)}\n")

VARI_mean_results .close()
