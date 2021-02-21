import numpy as np
import matplotlib.pyplot as plt
from utilities import calculate_optimal_threshold, calculate_VI_RGB, calculate_VI_fraction, read_data
from sklearn.metrics import f1_score
import time
from sklearn.linear_model import LinearRegression
from opt_parameters import min_plant_size
from skimage.morphology import remove_small_objects, remove_small_holes
from fraction_indicator_PSO_optimatization import Particle, Space

def choose_pixels_with_mask(bands=None, ground_truth=None, mask=None, is_RGB_map=None):
    X = []
    Y = []
    x_values, y_values = np.where(mask == 1)
    no_values = x_values.shape[0]
    for i in range(no_values):
        x = x_values[i]
        y = y_values[i]
        if is_RGB_map:
            X.append(bands["RGB"][x, y])
        else:
            X.append(bands[x, y])
        Y.append(ground_truth[x, y])
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

path_to_data = "data//Segmentation//"
dates = ["2019_07_25", "2019_09_20", "2019_10_11"]
use_morphological_operation = 1

bands_train, bands_test, ground_truth_train, ground_truth_test, mask_train, mask_test = read_data(dates=dates)

X_train = dict()
Y_train = dict()
for date in dates:
    X_train[date], Y_train[date] = choose_pixels_with_mask(
        bands=bands_train[date], ground_truth=ground_truth_train[date], mask=mask_train[date], is_RGB_map=1)

if not use_morphological_operation:

    X_test = dict()
    Y_test = dict()
    for date in dates:
        X_test[date], Y_test[date] = choose_pixels_with_mask(
            bands=bands_test[date], ground_truth=ground_truth_test[date], mask=mask_test[date], is_RGB_map=1)

else:
    X_test = dict()
    Y_test = dict()
    for date in dates:
        X_test[date] = bands_test[date]["RGB"]
        Y_test[date] = ground_truth_test[date]

results_segmentation = open("Results_segmentation.txt", "w")
results_segmentation.write("date\tclf_type\tF1_score\ttraining_time\tinference_time\tuse_morphological_operation\n")
results_segmentation.close()

print("1. Standard Vegetation Indicators")

VI_RGB_train = dict()
VI_RGB_test = dict()
threshold_VI = dict()

for date in dates:

    VI_RGB_train[date] = calculate_VI_RGB(R=X_train[date][:, 0], G=X_train[date][:, 1], B=X_train[date][:, 2])
    if not use_morphological_operation:
        VI_RGB_test[date] = calculate_VI_RGB(R=X_test[date][:, 0], G=X_test[date][:, 1], B=X_test[date][:, 2])
    else:
        VI_RGB_test[date] = calculate_VI_RGB(R=X_test[date][:, :, 0], G=X_test[date][:, :, 1], B=X_test[date][:, :, 2])
    threshold_VI[date] = dict()

    for VI_name in VI_RGB_train[date].keys():

        start_training_time = time.time()

        mean_ = np.mean(VI_RGB_train[date][VI_name])
        std_ = np.std(VI_RGB_train[date][VI_name])
        threshold_start = mean_ - 1.5 * std_
        threshold_end = mean_ + 1.5 * std_

        threshold_VI[date][VI_name], F1_max = calculate_optimal_threshold(
            VI=VI_RGB_train[date][VI_name], Y_true=Y_train[date], repeats=5, n=20,
            threshold_start=threshold_start, threshold_end=threshold_end)

        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        training_time = np.round(training_time, 2)

        start_inference_time = time.time()

        Y_predict_test = VI_RGB_test[date][VI_name]
        Y_predict_test = (Y_predict_test > threshold_VI[date][VI_name])
        if use_morphological_operation:
            Y_predict_test = remove_small_holes(Y_predict_test, area_threshold=min_plant_size[date])
            Y_predict_test = remove_small_objects(Y_predict_test, min_size=min_plant_size[date])

        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
        inference_time = np.round(inference_time, 2)

        if use_morphological_operation:
            Y_predict_test, Y_test[date] = choose_pixels_with_mask(
                bands=Y_predict_test, ground_truth=ground_truth_test[date], mask=mask_test[date], is_RGB_map=False)

        F1_score = f1_score(Y_predict_test, Y_test[date])
        F1_score = np.round(F1_score, 4)

        print(f"{date}\t{VI_name}\t{F1_score}\t{training_time}\t{inference_time}\t{use_morphological_operation}\n")
        results_segmentation = open("Results_segmentation.txt", "a")
        results_segmentation.write(
            f"{date}\t{VI_name}\t{F1_score}\t{training_time}\t{inference_time}\t{use_morphological_operation}\n")
        results_segmentation.close()

print("2. Optimized Linear Indicators")

for date in dates:

    start_training_time = time.time()

    clf = LinearRegression().fit(X_train[date], Y_train[date])

    Y_predict_train = clf.predict(X_train[date])
    threshold_opt, _ = calculate_optimal_threshold(
        VI=Y_predict_train, Y_true=Y_train[date], repeats=5, n=20, threshold_start=0, threshold_end=1)

    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    training_time = np.round(training_time, 2)

    start_inference_time = time.time()

    if not use_morphological_operation:
        Y_predict_test = clf.predict(X_test[date])
        Y_predict_test = (Y_predict_test > threshold_opt)
    if use_morphological_operation:
        rows, cols, channels = X_test[date].shape
        X_test_temp = np.reshape(X_test[date], newshape=(rows * cols, channels))
        Y_predict_test = clf.predict(X_test_temp)
        Y_predict_test = (Y_predict_test > threshold_opt)
        Y_predict_test = np.reshape(Y_predict_test, newshape=(rows, cols))
        Y_predict_test = remove_small_holes(Y_predict_test, area_threshold=min_plant_size[date])
        Y_predict_test = remove_small_objects(Y_predict_test, min_size=min_plant_size[date])

    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    inference_time = np.round(inference_time, 2)

    if use_morphological_operation:
        Y_predict_test, Y_test[date] = choose_pixels_with_mask(
            bands=Y_predict_test, ground_truth=ground_truth_test[date], mask=mask_test[date], is_RGB_map=False)

    F1_score = f1_score(Y_predict_test, Y_test[date])
    F1_score = np.round(F1_score, 4)

    print(f"{date}\tlinear\t{F1_score}\t{training_time}\t{inference_time}\t{use_morphological_operation}\n")
    results_segmentation = open("Results_segmentation.txt", "a")
    results_segmentation.write(
        f"{date}\tlinear\t{F1_score}\t{training_time}\t{inference_time}\t{use_morphological_operation}\n")
    results_segmentation.close()

print("3. Optimized Fraction Indicators")

for date in dates:

    start_training_time = time.time()

    n_iterations = 50
    n_particles = 30
    search_space = Space(n_particles, X_train[date], Y_train[date])
    particles_vector = [Particle() for _ in range(search_space.n_particles)]
    search_space.particles = particles_vector

    iteration = 0
    while(iteration < n_iterations):
        search_space.set_pbest()
        search_space.move_particles()
        iteration += 1

    coefs_best = np.array(search_space.gbest_position)
    w_up = coefs_best[:3]
    w_down = coefs_best[3:]

    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    training_time = np.round(training_time, 2)

    start_inference_time = time.time()

    if not use_morphological_operation:
        VI_test = calculate_VI_fraction(X=X_test[date], w_down=w_down, w_up=w_up)
        Y_predict_test = (VI_test > 0.2)
    else:
        rows, cols, channels = X_test[date].shape
        X_test_temp = np.reshape(X_test[date], newshape=(rows * cols, channels))
        Y_predict_test = calculate_VI_fraction(X=X_test_temp, w_down=w_down, w_up=w_up)
        Y_predict_test = (Y_predict_test > 0.2)
        Y_predict_test = np.reshape(Y_predict_test, newshape=(rows, cols))
        Y_predict_test = remove_small_holes(Y_predict_test, area_threshold=min_plant_size[date])
        Y_predict_test = remove_small_objects(Y_predict_test, min_size=min_plant_size[date])

    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    inference_time = np.round(inference_time, 2)

    if use_morphological_operation:
        Y_predict_test, Y_test[date] = choose_pixels_with_mask(
            bands=Y_predict_test, ground_truth=ground_truth_test[date], mask=mask_test[date], is_RGB_map=False)

    F1_score = f1_score(Y_predict_test, Y_test[date])
    F1_score = np.round(F1_score, 4)

    print(f"{date}\tfraction\t{F1_score}\t{training_time}\t{inference_time}\t{use_morphological_operation}\n")
    results_segmentation = open("Results_segmentation.txt", "a")
    results_segmentation.write(
        f"{date}\tfraction\t{F1_score}\t{training_time}\t{inference_time}\t{use_morphological_operation}\n")
    results_segmentation.close()
