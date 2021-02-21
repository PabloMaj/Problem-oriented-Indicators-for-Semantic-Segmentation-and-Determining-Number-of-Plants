import numpy as np
from scipy.ndimage import label
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from tqdm import tqdm
from numba import jit
import time
from utilities import calculate_optimal_threshold, calculate_VI_fraction, calculate_F1, read_data
from fraction_indicator_PSO_optimatization import Particle, Space

def create_test_samples(dates=None, bands_test=None, ground_truth_test=None, mask_test=None):
    X_test = dict()
    Y_test = dict()
    for date in dates:
        X_test[date] = []
        Y_test[date] = []
        x_values, y_values = np.where(mask_test[date] == 1)
        for pixel_id in range(0, x_values.shape[0]):
            x = x_values[pixel_id]
            y = y_values[pixel_id]
            R_test = bands_test[date]["Red"][x, y]
            G_test = bands_test[date]["Green"][x, y]
            B_test = bands_test[date]["Blue"][x, y]
            X_test[date].append([R_test, G_test, B_test])
            Y_test[date].append(ground_truth_test[date][x, y])
        X_test[date] = np.array(X_test[date])
        Y_test[date] = np.array(Y_test[date])

    return X_test, Y_test

def create_train_samples_0_label(dates=None, bands_train=None, ground_truth_train=None, mask_train=None):
    X_train_0_label = dict()
    Y_train_0_label = dict()
    for date in dates:
        X_train_0_label[date] = []
        Y_train_0_label[date] = []
        x_values, y_values = np.where(mask_train[date] == 1)
        for pixel_id in tqdm(range(0, x_values.shape[0])):
            x = x_values[pixel_id]
            y = y_values[pixel_id]
            if ground_truth_train[date][x, y] == 0:
                R_train = bands_train[date]["Red"][x, y]
                G_train = bands_train[date]["Green"][x, y]
                B_train = bands_train[date]["Blue"][x, y]
                X_train_0_label[date].append([R_train, G_train, B_train])
                Y_train_0_label[date].append(ground_truth_train[date][x, y])

    return X_train_0_label, Y_train_0_label

def create_train_samples(
        bands_train=None, ground_truth_train=None, X_train_0_label=None, Y_train_0_label=None, blob_filtered=None):

    x_values, y_values = np.where(blob_filtered == 1)
    X_train = []
    Y_train = []
    for pixel_id in range(0, x_values.shape[0]):
        x = x_values[pixel_id]
        y = y_values[pixel_id]
        if mask_train[date][x, y] == 1:
            R_train = bands_train[date]["Red"][x, y]
            G_train = bands_train[date]["Green"][x, y]
            B_train = bands_train[date]["Blue"][x, y]
            X_train.append([R_train, G_train, B_train])
            Y_train.append(ground_truth_train[date][x, y])
    X_train += X_train_0_label[date]
    Y_train += Y_train_0_label[date]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train


dates = ['2019_07_25', '2019_09_20', '2019_10_11']
N = 200
repeats = 5
version = "fraction"

print("1. Read data")
bands_train, bands_test, ground_truth_train, ground_truth_test, mask_train, mask_test = read_data(dates=dates)

print("2. Create test samples")
X_test, Y_test = create_test_samples(
    dates=dates, bands_test=bands_test, ground_truth_test=ground_truth_test, mask_test=mask_test)

print("3. Create train samples with 0 label")
X_train_0_label, Y_train_0_label = create_train_samples_0_label(
    dates=dates, bands_train=bands_train, ground_truth_train=ground_truth_train, mask_train=mask_train)

print("4. Create output files")
if version == "linear":
    results_file = open(f"Results_f1_score_vs_no_samples_{version}.txt", "a")
    results_file.write("date\tn\trepeat\tF1_score_test\tcoefs[0]\tcoefs[1]\tcoefs[2]\tintercept\tthreshold\n")
elif version == "fraction":
    results_file = open(f"Results_f1_score_vs_no_samples_{version}.txt", "a")
    results_file.write("date\tn\trepeat\tF1_score_test\t")
    for i in range(0, 6):
        results_file.write(f"coefs[{i}]")
        if i == 5:
            results_file.write("\n")
        else:
            results_file.write("\t")

print("5. Training")
for date in dates:

    blobs_GT, num_blobs = label(ground_truth_train[date])  # Labeling blobs in GT train

    for n in [200]:  # range(1, (N+1)): #number of labeled plants

        F1_scores = []

        for repeat in range(0, repeats):  # repeats

            blob_labels = random.sample(list(range(1, num_blobs + 1)), n)  # random choose n plants
            blob_filtered = np.isin(blobs_GT, blob_labels)
            X_train, Y_train = create_train_samples(
                bands_train=bands_train, ground_truth_train=ground_truth_train,
                X_train_0_label=X_train_0_label, Y_train_0_label=Y_train_0_label, blob_filtered=blob_filtered)

            if version == "linear":

                clf = LinearRegression().fit(X_train, Y_train)

                Y_predict_train = clf.predict(X_train)
                threshold_opt, _ = calculate_optimal_threshold(
                    VI=Y_predict_train, Y_true=Y_train, repeats=5, n=20, threshold_start=0, threshold_end=1)

                Y_predict_test = clf.predict(X_test[date])
                Y_predict_test = (Y_predict_test > threshold_opt)

                F1_score_test = f1_score(Y_test[date], Y_predict_test)
                F1_scores.append(F1_score_test)

                coefs = clf.coef_
                intercept = clf.intercept_
                print(f"{date}\t{n}\t{repeat}\t{F1_score_test}\t{threshold_opt}")

                results_file = open(f"Results_f1_score_vs_no_samples_{version}.txt", "a")
                results_file.write(f"{date}\t{n}\t{repeat}\t{F1_score_test}\t \
                {coefs[0]}\t{coefs[1]}\t{coefs[2]}\t{intercept}\t{threshold_opt}\n")
                results_file.close()

            elif version == "fraction":

                n_iterations = 20
                n_particles = 30
                search_space = Space(n_particles, X_train, Y_train)
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

                VI_test = calculate_VI_fraction(X=X_test[date], w_down=w_down, w_up=w_up)
                Y_predict_test = (VI_test > 0.2)

                F1_score_test = f1_score(Y_test[date], Y_predict_test)
                F1_scores.append(F1_score_test)

                print(f"{date}\t{n}\t{repeat}\t{F1_score_test}")
                results_file = open(f"Results_f1_score_vs_no_samples_{version}.txt", "a")
                results_file.write(f"{date}\t{n}\t{repeat}\t{F1_score_test}\t")
                for i in range(0, 6):
                    results_file.write(f"{coefs_best[i]}")
                    if i == 5:
                        results_file.write("\n")
                    else:
                        results_file.write("\t")
                results_file.close()

        F1_scores_mean = np.mean(F1_scores)
        F1_scores_std = np.std(F1_scores)
        print(f"F1_score={F1_scores_mean}+-{F1_scores_std}")
