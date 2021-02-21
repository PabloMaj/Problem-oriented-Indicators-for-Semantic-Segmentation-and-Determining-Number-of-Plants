import numpy as np
from sklearn.metrics import f1_score
from numba import jit

# Calculate optimal threshold for calculated indicator
def calculate_optimal_threshold(VI=None, Y_true=None, repeats=5, n=20, threshold_start=0, threshold_end=1):
    for _ in range(repeats):
        results = []
        for threshold in np.linspace(start=threshold_start, stop=threshold_end, num=n):
            results.append((threshold, f1_score((VI > threshold), Y_true)))
        item_max = max(results, key=lambda x: x[1])
        threshold_opt = item_max[0]
        F1_max = item_max[1]
        pos_max = results.index(item_max)
        pos_max = min(n - 2, pos_max)
        pos_max = max(1, pos_max)
        threshold_start = results[pos_max - 1][0]
        threshold_end = results[pos_max + 1][0]
    return threshold_opt, F1_max

# Calculate fraction indicator based on given parameters
@jit(nopython=True, parallel=True)
def calculate_VI_fraction(X=None, w_up=None, w_down=None):
    epsilon = 10**-6
    VI = np.sum(X * w_up, axis=1) / (np.sum(X * w_down, axis=1) + epsilon)
    return VI

# Calculate F1-score between thresholded indicator and ground truth
@jit(nopython=True, parallel=True)
def calculate_F1(VI=None, threshold=0.5, ground_truth=None):
    mask_predicted = (VI > threshold)
    TP = np.multiply((mask_predicted == ground_truth), ground_truth == 1)
    FP = np.multiply((mask_predicted != ground_truth), ground_truth == 1)
    FN = np.multiply((mask_predicted != ground_truth), ground_truth == 0)
    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)
    F1_score = 2 * TP / (2 * TP + FP + FN)
    return F1_score

def calculate_VI_RGB(R=None, G=None, B=None):
    R = R.astype('float')
    G = G.astype('float')
    B = B.astype('float')
    VI_RGB = dict()
    VI_RGB['ExG'] = 2 * G - R - B
    VI_RGB["TGI"] = G - 0.39 * R - 0.61 * B
    VI_RGB['CIVE'] = -(0.441 * R - 0.881 * G + 0.385 * B + 18.787)
    VI_RGB['RGRI'] = -R / G
    VI_RGB['NGRDI'] = (G - R) / (G + R)
    VI_RGB['VARI'] = (G - R) / (G + R - B)
    VI_RGB['VDVI'] = (2 * G - R - B) / (2 * G + R + B)
    VI_RGB['VEG'] = G / (np.power(R, 0.667) * np.power(B, 1 - 0.667))
    VI_RGB['MGRVI'] = (G**2 - R**2) / (G**2 + R**2)
    VI_RGB["RGBVI"] = (G**2 - B * R) / (G**2 + B * R)
    for VI in VI_RGB:
        VI_RGB[VI] = np.nan_to_num(VI_RGB[VI])
    return VI_RGB

def read_data(dates=None, path_to_data="data//Segmentation//"):

    bands_train = dict()
    ground_truth_train = dict()
    mask_train = dict()

    bands_test = dict()
    ground_truth_test = dict()
    mask_test = dict()

    for date in dates:

        bands_train[date] = dict()
        bands_test[date] = dict()

        for channel in ["Red", "Green", "Blue"]:

            bands_train[date][channel] = np.load(path_to_data + date + '_train_' + channel + '.npy')
            bands_test[date][channel] = np.load(path_to_data + date + '_test_' + channel + '.npy')

        bands_train[date]["RGB"] = np.dstack(
            (bands_train[date]["Red"], bands_train[date]["Green"], bands_train[date]["Blue"]))

        bands_test[date]["RGB"] = np.dstack(
            (bands_test[date]["Red"], bands_test[date]["Green"], bands_test[date]["Blue"]))

        ground_truth_train[date] = np.load(path_to_data + date + '_train_GT.npy')
        ground_truth_test[date] = np.load(path_to_data + date + '_test_GT.npy')

        mask_train[date] = np.load(path_to_data + date + '_train_mask.npy')
        mask_test[date] = np.load(path_to_data + date + '_test_mask.npy')

    return bands_train, bands_test, ground_truth_train, ground_truth_test, mask_train, mask_test
