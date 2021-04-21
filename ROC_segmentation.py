import numpy as np
import matplotlib.pyplot as plt
from opt_parameters import clf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
plt.rcParams.update({'font.size': 14})

def read_data(dates=None, path_to_data="data//"):

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

def choose_pixels_with_mask(VI_predicted=None, ground_truth=None, mask=None):
    X = []
    Y = []
    x_values, y_values = np.where(mask == 1)
    no_values = x_values.shape[0]
    for i in range(no_values):
        x = x_values[i]
        y = y_values[i]
        label = ground_truth[x, y]
        if label == 0:
            X.append(VI_predicted[x, y])
        elif label == 1:
            X.append(VI_predicted[x, y])
        Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

dates = ['2019_07_25', '2019_09_20', '2019_10_11']
clf_types = ['ExG', 'Linear', 'VDVI', 'Fraction']
path_to_data = "data//Segmentation//"

bands_test, ground_truth_test, mask_test = read_data(dates=dates, path_to_data=path_to_data)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

x_lims = [0.25, 0.5, 0.75]

results_ROC = open("Results_ROC.txt", "w")
results_ROC.write("date\tclf_type\tAUC\n")

labels = ["flowering", "mature", "before harvest"]

for counter_date, date in enumerate(dates):
    GT = ground_truth_test[date]
    for counter_clf, clf_type in enumerate(clf_types):

        R = bands_test[date]["Red"]
        G = bands_test[date]["Green"]
        B = bands_test[date]["Blue"]

        if clf_type == "ExG":
            VI_predicted = 2 * G - R - B
            VI_predicted = np.nan_to_num(VI_predicted)

        if clf_type == "VDVI":
            VI_predicted = (2 * G - R - B) / (2 * G + R + B)

        if clf_type == "Linear":
            coefs = clf[date][clf_type]['coefs']
            VI_predicted = coefs[0] * R + coefs[1] * G + coefs[2] * B + coefs[3]

        if clf_type == "Fraction":
            coefs = clf[date][clf_type]['coefs']
            VI_predicted = (coefs[0] * R + coefs[1] * G + coefs[2] * B) / (coefs[3] * R + coefs[4] * G + coefs[5] * B)

        X, Y = choose_pixels_with_mask(VI_predicted=VI_predicted, ground_truth=ground_truth_test[date], mask=mask_test[date])
        XY = zip(X, Y)
        XY_sorted = sorted(XY)
        XY_tuples = zip(*XY_sorted)
        X, Y = [list(XY_tuple) for XY_tuple in XY_tuples]
        fpr, tpr, thresholds = metrics.roc_curve(Y, X, pos_label=1)
        axs[counter_date].plot(fpr, tpr)
        axs[counter_date].set_xlabel("FPR")
        axs[counter_date].set_ylabel("TPR")
        axs[counter_date].set_title(f"{labels[counter_date]}")
        axs[counter_date].grid(color='b', ls='-.', lw=0.25)
        axs[counter_date].set_xlim(-0.05, 1)
        axs[counter_date].set_ylim(0, 1.05)
        axs[counter_date].legend(['ExG', 'Optimised linear', 'VDVI', 'Optimised fraction'], loc="lower right")
        axs[counter_date].set_xlim(0, x_lims[counter_date])
        AUC = roc_auc_score(Y, X)
        results_ROC.write(f"{date}\t{clf_type}\t{np.round(AUC, 4)}\n")

results_ROC.close()
plt.savefig("ROC_curves.png", dpi=300)
