import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion, binary_closing, binary_opening, remove_small_objects, remove_small_holes
import time
from opt_parameters import clf, min_plant_size
from sklearn.linear_model import LinearRegression
import warnings

warnings. filterwarnings('ignore')

def calculate_mask_predicted(clf_type=None, clf=clf, bands=None, mask=None, min_plant_size=None, read_Mask_RCNN=False):

    R = bands["Red"]
    G = bands["Green"]
    B = bands["Blue"]

    if clf_type == "ExG":
        mask_predicted_ = 2 * G - R - B
        mask_predicted_ = (mask_predicted_ > clf[date][clf_type]['threshold'])
        plt.show()

    if clf_type == "VDVI":
        mask_predicted_ = (2 * G - R - B) / (2 * G + R + B)
        mask_predicted_ = (mask_predicted_ > clf[date][clf_type]['threshold'])

    if clf_type == "Linear":
        coefs = clf[date][clf_type]['coefs']
        mask_predicted_ = coefs[0] * R + coefs[1] * G + coefs[2] * B + coefs[3]
        mask_predicted_ = (mask_predicted_ > clf[date][clf_type]['threshold'])

    if clf_type == "Fraction":
        coefs = clf[date][clf_type]['coefs']
        mask_predicted_ = (coefs[0] * R + coefs[1] * G + coefs[2] * B) / (coefs[3] * R + coefs[4] * G + coefs[5] * B)
        mask_predicted_ = (mask_predicted_ > 0.2)

    if clf_type == "Mask_RCNN" and read_Mask_RCNN:
        mask_predicted_ = np.load(date.replace("_", "-") + "_Mask_RCNN_output.npy")

    mask_predicted_ = np.multiply(mask_predicted_, mask)
    mask_predicted_ = mask_predicted_.astype('int')
    mask_predicted_ = remove_small_holes(mask_predicted_, area_threshold=20)
    mask_predicted_ = remove_small_objects(mask_predicted_, min_size=min_plant_size[date])

    return mask_predicted_

dates = ['2019_07_25', '2019_09_20', '2019_10_11']
clf_types = ['ExG', 'Linear', 'VDVI', 'Fraction']
bands_names = ["Red", "Green", "Blue"]
path_to_data = "data//Counting//"

data_train = dict()
data_test = dict()
mask_train = dict()
mask_test = dict()

# Read data

for date in dates:

    data_train[date] = dict()
    data_test[date] = dict()
    mask_train[date] = dict()
    mask_test[date] = dict()

    # Train dataset
    for train_no in [10, 20, 40, 60, 80, 100, 200, 300, 440]:
        data_train[date][train_no] = dict()
        for band_name in bands_names:
            data_train[date][train_no][band_name] = np.load(path_to_data + "Train//" + f"{date}_{band_name}_id_{train_no}_counting.npy")
            # print(data_train[date][train_no][band_name].shape)
        mask_train[date][train_no] = np.load(path_to_data + "Train//" + f"{date}_mask_id_{train_no}_counting.npy")
        # print(mask_train[date][train_no].shape)

    # Test dataset
    for band_name in bands_names:
        data_test[date][band_name] = np.load(path_to_data + "Test//" + f"{date}_{band_name}_counting.npy")
        # print(data_test[date][band_name].shape)
    mask_test[date] = np.load(path_to_data + "Test//" + f"{date}_mask_counting.npy")
    # print(mask_test[date].shape)

results_counting = open("Results_counting.txt", "w")
results_counting.write("date\tclf_type\tMAPE\ttraining_time\tinference_time\n")

# linear regression model - dates: '2019_09_20', '2019_10_11'
canopy_size = dict()
models_regression = dict()
for date in dates[1:]:
    models_regression[date] = dict()
    canopy_size[date] = dict()
    for clf_type in clf_types:

        start_training_time = time.time()
        canopy_size[date][clf_type] = []
        for train_no in [10, 20, 40, 60, 80, 100, 200, 300, 440]:
            mask_predicted_ = calculate_mask_predicted(clf_type=clf_type, clf=clf, bands=data_train[date][train_no], mask=mask_train[date][train_no], min_plant_size=min_plant_size)
            canopy_size[date][clf_type].append((train_no, np.sum(mask_predicted_)))
        x = np.array([el[1] for el in canopy_size[date][clf_type]]).reshape(-1, 1)
        y = np.array([el[0] for el in canopy_size[date][clf_type]])
        models_regression[date][clf_type] = LinearRegression().fit(x, y)
        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        training_time = np.round(training_time, 2)

        # print(f"Data:{date} Klasyfikator:{clf_type} R^2:{models_regression[date][clf_type].score(x, y)}")

        start_inference_time = time.time()
        mask_predicted_ = calculate_mask_predicted(clf_type=clf_type, clf=clf, bands=data_test[date], mask=mask_test[date], min_plant_size=min_plant_size)
        x = np.array([np.sum(mask_predicted_)]).reshape(-1, 1)
        y_predict = models_regression[date][clf_type].predict(x)[0]
        end_inference_time = time.time()
        time_inference = end_inference_time - start_inference_time
        time_inference = np.round(time_inference, 2)

        y_true = 400
        MAPE = np.round(np.abs(y_predict - y_true) * 100 / y_true, 2)

        results_counting.write(f"{date}\t{clf_type}\t{MAPE}\t{training_time}\t{time_inference}\n")
        print(f"{date}\t{clf_type}\t{MAPE}\t{training_time}\t{time_inference}\n")

# erosion tuning - dates: '2019_07_25'
erosion_opt_iterations = dict()
for date in dates[:1]:
    erosion_opt_iterations[date] = dict()
    for clf_type in clf_types:

        min_error = 1000
        opt_no = None
        y_true_train = 440
        y_true_test = 400

        start_training_time = time.time()
        mask_predicted_ = calculate_mask_predicted(clf_type=clf_type, clf=clf, bands=data_train[date][y_true_train], mask=mask_train[date][y_true_train], min_plant_size=min_plant_size)
        for erosion_iterations in range(0, 10):
            mask_predicted_ = binary_erosion(mask_predicted_)
            for _ in range(erosion_iterations):
                mask_predicted_ = binary_erosion(mask_predicted_)
            mask_predicted_ = remove_small_objects(mask_predicted_, min_size=min_plant_size[date])
            labels, num = label(mask_predicted_, return_num=True)
            Y_predicted = num
            error = (y_true_train - Y_predicted)
            if np.abs(error) < min_error:
                min_error = np.abs(error)
                error_with_sign = error
                opt_no = erosion_iterations
                labels_opt = labels
                mask_predicted_opt = mask_predicted_
        erosion_opt_iterations[date][clf_type] = opt_no
        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        training_time = np.round(training_time, 2)

        start_time_inference = time.time()
        mask_predicted_ = calculate_mask_predicted(clf_type=clf_type, clf=clf, bands=data_test[date], mask=mask_test[date], min_plant_size=min_plant_size)
        for i in range(erosion_opt_iterations[date][clf_type]):
            mask_predicted_ = binary_erosion(mask_predicted_)
        mask_predicted_ = remove_small_objects(mask_predicted_, min_size=min_plant_size[date])
        labels, num = label(mask_predicted_, return_num=True)
        y_predict = num
        end_time_inference = time.time()
        time_inference = end_time_inference - start_time_inference
        time_inference = np.round(time_inference, 2)

        y_true_test = 400
        MAPE = np.round(np.abs(y_predict - y_true_test) * 100 / y_true_test, 2)

        results_counting.write(f"{date}\t{clf_type}\t{MAPE}\t{training_time}\t{time_inference}\n")
        print(f"{date}\t{clf_type}\t{MAPE}\t{training_time}\t{time_inference}\n")

results_counting.close()
