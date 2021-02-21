import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

dates = ['2019_07_25', '2019_09_20', '2019_10_11']
colors = ["red", "green", "blue", "orange"]

for version in ['linear', 'fraction']:

    plt.clf()
    df = pd.read_csv(f"Results_f1_score_vs_no_samples_{version}.txt", sep="\t").dropna()

    for counter, date in enumerate(dates):
        df_filter_date = df[df['date'] == date]
        no_of_samples = np.unique(list(df_filter_date['n']))
        x = []
        y_mean = []
        y_std = []
        for n in no_of_samples:
            df_filter_n = df_filter_date[df_filter_date['n'] == n]
            y_values = list(df_filter_n['F1_score_test'].values)
            y_mean.append(np.mean(y_values))
            y_std.append(np.std(y_values))
            x.append(n)
        y_mean = np.array(y_mean)
        y_mean = savgol_filter(y_mean, 31, 3)
        y_std = np.array(y_std)
        y_std = savgol_filter(y_std, 31, 3)
        plt.plot(x, y_mean, color=colors[counter])
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=colors[counter], alpha=0.2)
    plt.grid()
    plt.title(f"{version[0].upper()}{version[1:]} model type")
    plt.legend(dates, loc="lower right")
    plt.xlabel("Number of labeled samples")
    plt.ylabel("F1-score")
    plt.savefig(f"f1_score_vs_no_samples_{version}.png", dpi=300)
    # plt.show()
