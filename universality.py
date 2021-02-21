import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from utilities import calculate_optimal_threshold, calculate_VI_RGB, calculate_VI_fraction

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
            Y_predict_test = (Y_predict_test > threshold_VI[train_date][VI_name])
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
        Y_predict_test = (Y_predict_test > threshold[train_date])
        F1_score = f1_score(Y_predict_test, Y_test[test_date])
        print(f"Train_date:{train_date}")
        print(f"Test_date:{test_date}")
        print(f"F1_score:{np.round(F1_score,4)}")
        results_file = open("Results_robustness.txt", "a")
        results_file.write(f"linear_optimized\t{train_date}\t{test_date}\t{np.round(F1_score,4)}\n")
        results_file.close()

print("3. Optimized Fraction Indicators")

VI_fraction_coeffs = dict()
VI_fraction_coeffs["2019_07_25"] = [36.80527083, -64.29557348, 30.29631855, -20.51248141, -17.49035232, -88.2301978]
VI_fraction_coeffs["2019_08_19"] = [5.60702874, -7.99014732, 5.33548688, -0.77724135, -3.57582202, -0.10127517]
VI_fraction_coeffs["2019_09_20"] = [26.71492253, -34.8274881, 23.80132208, 11.11393239, -20.81248397, 8.17381451]
VI_fraction_coeffs["2019_10_11"] = [0.08974777, -3.95925329, 1.19164462, 0.52334907, -1.0482393, 1.24637182]
for date in dates:
    VI_fraction_coeffs[date] = np.array(VI_fraction_coeffs[date])

for train_date in dates:
    for test_date in dates:
        Y_predict_test = calculate_VI_fraction(
            X=X_test[test_date], w_up=VI_fraction_coeffs[train_date][:3], w_down=VI_fraction_coeffs[train_date][3:])
        F1_score = f1_score(Y_predict_test > 0.2, Y_test[test_date])
        print(f"Train_date:{train_date}")
        print(f"Test_date:{test_date}")
        print(f"F1_score:{np.round(F1_score, 4)}")
        results_file = open("Results_robustness.txt", "a")
        results_file.write(f"fraction_optimized\t{train_date}\t{test_date}\t{np.round(F1_score, 4)}\n")
        results_file.close()
