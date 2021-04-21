import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from utilities import calculate_optimal_threshold


path_to_data = "data//Segmentation//"
dates = ["2019_07_25", "2019_09_20", "2019_10_11"]
test_other_models = False

for date in dates:

    X_train = np.load(path_to_data + 'X_train_' + date + '_add_Color_Space.npy')
    X_test = np.load(path_to_data + 'X_test_' + date + '_add_Color_Space.npy')
    Y_train = np.load(path_to_data + 'Y_train_' + date + '_add_Color_Space.npy')
    Y_test = np.load(path_to_data + 'Y_test_' + date + '_add_Color_Space.npy')

    X_train_all_features = X_train
    X_test_all_features = X_test

    models = [(LinearRegression(), "LinReg")]
    if test_other_models:
        models.append((LogisticRegression(), "LogReg"))
        models.append((MLPClassifier(hidden_layer_sizes=(5, 5), verbose=True), "MLP"))

    color_spaces = ["RGB", "Lab", "HSV", "RGB_Lab", "RGB_HSV", "Lab_HSV", "RGB_HSV_Lab"]

    for clf, clf_name in models:
        for color_space in color_spaces:
            if color_space == "RGB":
                X_train = X_train_all_features[:, :3]
                X_test = X_test_all_features[:, :3]
            elif color_space == "Lab":
                X_train = X_train_all_features[:, 3:6]
                X_test = X_test_all_features[:, 3:6]
            elif color_space == "HSV":
                X_train = X_train_all_features[:, 6:9]
                X_test = X_test_all_features[:, 6:9]
            elif color_space == "RGB_Lab":
                X_train = X_train_all_features[:, :6]
                X_test = X_test_all_features[:, :6]
            elif color_space == "RGB_HSV":
                X_train = np.concatenate([X_train_all_features[:, :3], X_train_all_features[:, 6:9]], axis=1)
                X_test = np.concatenate([X_test_all_features[:, :3], X_test_all_features[:, 6:9]], axis=1)

            elif color_space == "Lab_HSV":
                X_train = X_train_all_features[:, 3:9]
                X_test = X_test_all_features[:, 3:9]
            elif color_space == "RGB_HSV_Lab":
                X_train = X_train_all_features
                X_test = X_test_all_features

            clf.fit(X_train, Y_train)

            if clf_name == "LinReg":
                Y_predict_train = clf.predict(X_train)
                Y_predict_test = clf.predict(X_test)
            elif clf_name == "LogReg":
                Y_predict_train = clf.predict_proba(X_train)[:, 1]
                Y_predict_test = clf.predict_proba(X_test)[:, 1]
            elif clf_name == "MLP":
                Y_predict_train = clf.predict(X_train)
                Y_predict_test = clf.predict(X_test)

            if clf_name in ["LinReg", "LogReg"]:
                threshold_opt, F1_train = calculate_optimal_threshold(VI=Y_predict_train, Y_true=Y_train)
                F1_test = f1_score((Y_predict_test > threshold_opt), Y_test)

            elif clf_name in ["MLP"]:
                F1_train = f1_score(Y_predict_train, Y_train)
                F1_test = f1_score(Y_predict_test, Y_test)

            print("Date:" + date)
            print('Color_space:' + color_space)
            print("Classifer:" + clf_name)
            print(f"F1_train:{np.round(F1_train, 4)}")
            print(f"F1_test:{np.round(F1_test, 4)}")
            print(f"Threshold_opt:{np.round(threshold_opt, 4)}")
            print("--------------")

# visualization
labels = ["RGB", "Lab", "HSV", "RGB+Lab", "RGB+HSV", "Lab+HSV", "RGB+HSV+\nLab"]
x = np.arange(len(labels[1:]))
width = 0.2
y = dict()
y["2019_07_25"] = [0.8011, 0.8003, 0.7491, 0.8013, 0.8047, 0.8050, 0.8038]
y["2019_09_20"] = [0.7534, 0.7562, 0.7248, 0.7546, 0.7505, 0.7566, 0.7550]
y["2019_10_11"] = [0.8374, 0.8342, 0.7757, 0.8332, 0.8264, 0.8327, 0.8294]
for date in dates:
    y[date] = [el - y[date][0] for el in y[date][1:]]
fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width, y["2019_07_25"], width, label='flowering', zorder=3, edgecolor="k")
rects2 = ax.bar(x, y["2019_09_20"], width, label='before harvest', zorder=3, edgecolor="k")
rects3 = ax.bar(x + width, y["2019_10_11"], width, label='before harvest', zorder=3, edgecolor="k")
ax.set_xticks(x)
ax.set_xticklabels(labels[1:])
plt.grid(zorder=0)
plt.legend(["flowering", "mature", "before harvest"], loc="lower right", framealpha=1)
plt.ylabel("Change of F1-score compared to RGB")
plt.savefig("Add_color_spaces_testing.png", dpi=300)
