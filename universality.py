import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from utilities import calculate_optimal_threshold, calculate_VI_RGB, calculate_VI_fraction
from skimage.filters import threshold_otsu

def read_data(dates=None, path_to_data=None):

    X_train = dict()
    X_test = dict()
    Y_train = dict()
    Y_test = dict()

    for date in dates:

        X_train[date] = np.load(path_to_data + 'X_train_' + date + '_add_Color_Space.npy')[:, :3]
        X_test[date] = np.load(path_to_data + 'X_test_' + date + '_add_Color_Space.npy')[:, :3]
        Y_train[date] = np.load(path_to_data + 'Y_train_' + date + '_add_Color_Space.npy')
        Y_test[date] = np.load(path_to_data + 'Y_test_' + date + '_add_Color_Space.npy')

    return X_train, Y_train, X_test, Y_test

dates = ["2019_07_25", "2019_09_20", "2019_10_11"]
path_to_data = "data//Segmentation//"

results_file = open("Results_robustness.txt", "w")
results_file.write("Model_type\tTrain_date\tTest_date\tF1_score_test\n")
results_file.close()

X_train, Y_train, X_test, Y_test = read_data(dates=dates, path_to_data=path_to_data)

print("1. Standard Vegetation Indicators")

VI_RGB_train = dict()
VI_RGB_test = dict()
threshold_VI = dict()

for date in dates:

    VI_RGB_train[date] = calculate_VI_RGB(R=X_train[date][:, 0], G=X_train[date][:, 1], B=X_train[date][:, 2])
    VI_RGB_test[date] = calculate_VI_RGB(R=X_test[date][:, 0], G=X_test[date][:, 1], B=X_test[date][:, 2])
    threshold_VI[date] = dict()

    for VI_name in ["ExG", "VDVI"]:

        mean_ = np.mean(VI_RGB_train[date][VI_name])
        std_ = np.std(VI_RGB_train[date][VI_name])
        threshold_start = mean_ - 1.5 * std_
        threshold_end = mean_ + 1.5 * std_
        threshold_VI[date][VI_name], F1_max = calculate_optimal_threshold(
            VI=VI_RGB_train[date][VI_name], Y_true=Y_train[date], repeats=5, n=20,
            threshold_start=threshold_start, threshold_end=threshold_end)

for VI_name in ["ExG", "VDVI"]:
    for train_date in dates:
        for test_date in dates:
            Y_predict_test = VI_RGB_test[test_date][VI_name]

            thresh = threshold_otsu(image=Y_predict_test, nbins=1000)
            Y_predict_test = (Y_predict_test > thresh)

            # Y_predict_test = (Y_predict_test > threshold_VI[train_date][VI_name])

            F1_score = f1_score(Y_predict_test, Y_test[test_date])
            print(f"Train_date:{train_date}")
            print(f"Test_date:{test_date}")
            print(f"F1_score:{np.round(F1_score,4)}")
            results_file = open("Results_robustness.txt", "a")
            results_file.write(f"{VI_name}\t{train_date}\t{test_date}\t{np.round(F1_score,4)}\n")
            results_file.close()

print("2. Optimized Linear Indicators")

models = dict()
threshold = dict()
for train_date in dates:
    models[train_date] = LinearRegression().fit(X_train[train_date], Y_train[train_date])
    Y_predict_train = models[train_date].predict(X_train[train_date])
    threshold[train_date], _ = calculate_optimal_threshold(
        VI=Y_predict_train, Y_true=Y_train[train_date], repeats=5, n=20, threshold_start=0, threshold_end=1)

for train_date in dates:
    for test_date in dates:
        Y_predict_test = models[train_date].predict(X_test[test_date])

        thresh = threshold_otsu(image=Y_predict_test, nbins=1000)
        Y_predict_test = (Y_predict_test > thresh)

        # Y_predict_test = (Y_predict_test > threshold[train_date])

        F1_score = f1_score(Y_predict_test, Y_test[test_date])
        print(f"Train_date:{train_date}")
        print(f"Test_date:{test_date}")
        print(f"F1_score:{np.round(F1_score,4)}")
        results_file = open("Results_robustness.txt", "a")
        results_file.write(f"linear_optimized\t{train_date}\t{test_date}\t{np.round(F1_score,4)}\n")
        results_file.close()

print("3. Optimized Fraction Indicators")

VI_fraction_coeffs = dict()
VI_fraction_coeffs["2019_07_25"] = [-2.17140641, 3.70160468, -2.5102769, 2.88246843, 0.82232753, -1.68856676]
VI_fraction_coeffs["2019_09_20"] = [0.16585135, -0.78082026, 0.6213656, -1.07198113, -1.14975748, 0.7004156]
VI_fraction_coeffs["2019_10_11"] = [0.21016317, -2.23698845, 2.51774888, -2.93230797, -0.02953425, -2.48650525]
for date in dates:
    VI_fraction_coeffs[date] = np.array(VI_fraction_coeffs[date])

for train_date in dates:
    for test_date in dates:
        Y_predict_test = calculate_VI_fraction(
            X=X_test[test_date], w_up=VI_fraction_coeffs[train_date][:3], w_down=VI_fraction_coeffs[train_date][3:])

        thresh = threshold_otsu(image=Y_predict_test, nbins=1000)
        Y_predict_test = (Y_predict_test > thresh)

        F1_score = f1_score(Y_predict_test, Y_test[test_date])
        print(f"Train_date:{train_date}")
        print(f"Test_date:{test_date}")
        print(f"F1_score:{np.round(F1_score, 4)}")
        results_file = open("Results_robustness.txt", "a")
        results_file.write(f"fraction_optimized\t{train_date}\t{test_date}\t{np.round(F1_score, 4)}\n")
        results_file.close()

# visualization
dataset_names = {"2019_07_25": "flowering", "2019_09_20": "mature", "2019_10_11": "before harvest"}

fig, axs = plt.subplots(3, 3, figsize=(14, 14))
MaskRCNN_scores = [0.6613, 0.7013, 0.6316, 0.7758, 0.6341, 0.8412]
model_types = ["ExG", "VDVI", "Optimised\nlinear", "Optimised\nfraction", "Mask\nR-CNN"]
model_types = [model_types[i] for i in [0, 2, 1, 3, 4]]
df_results = pd.read_csv("Results_robustness.txt", sep="\t")
counter = 0
for train_counter, train_date in enumerate(dates):
    for test_counter, test_date in enumerate(dates):
        pos_1 = test_counter
        pos_2 = train_counter
        if train_date != test_date:
            df_filtered = df_results[(df_results["Train_date"] == train_date) & (df_results["Test_date"] == test_date)]
            y = list(df_filtered["F1_score_test"]) + [MaskRCNN_scores[counter]]
            axs[pos_1, pos_2].grid()
            y = [y[i] for i in [0, 2, 1, 3, 4]]
            print(y)
            axs[pos_1, pos_2].bar(model_types, y, color=["tab:blue", "tab:green", "orange", "plum", "tomato"], zorder=3, edgecolor="k")
            axs[pos_1, pos_2].set_ylim(0.6, 0.87)
            axs[pos_1, pos_2].set_ylabel("F1-score")
            axs[pos_1, pos_2].set_title(f"Train: {dataset_names[train_date]}\nTest: {dataset_names[test_date]}")
            counter += 1
        else:
            axs[pos_1, pos_2].set_xticks([])
            axs[pos_1, pos_2].set_yticks([])
            axs[pos_1, pos_2].text(0.5, 0.5, 'Figure 7',
            bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'none', 'pad': 1},
            ha='center', va='center', fontsize=20)
            axs[pos_1, pos_2].set_title(f"Train: {dataset_names[train_date]}\nTest: {dataset_names[test_date]}")


plt.tight_layout()
plt.savefig("Universality_results.png", dpi=300)
