# %%
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

from project import EXPERIMENTS_PATH
from experiments.plots import plot_mean_durations, plot_aucs
from experiments.metrics import calc_aucs

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "ionosphere_"
BASE_PATH = os.path.join(EXPERIMENTS_PATH, "classification", "incomplete")
FOLDER = os.path.join(BASE_PATH, name + "combined")
FOLDERS = glob(os.path.join(BASE_PATH, "*" + name + "*"))
FILENAMES = glob(os.path.join(FOLDERS[0], "csv", "*"))
FILENAMES = [os.path.basename(n) for n in FILENAMES]
CSV_FOLDER = os.path.join(FOLDER, "csv")

os.makedirs(FOLDER)
os.makedirs(CSV_FOLDER)

# %%
for filename in FILENAMES:
    for i, folder in enumerate(FOLDERS):
        path = os.path.join(folder, "csv", filename)
        if i == 0:
            df = pd.read_csv(path, index_col=0)
            if "_scores.csv" in filename:
                classifiers = list(df.columns)
                missing_rates = np.asarray(df.index)
            else:
                k_s = list(df.index)

        else:
            if "_scores.csv" in filename:
                continue
            df_new = pd.read_csv(path, index_col=0)
            df = pd.concat([df, df_new], axis=1)
    df.to_csv(os.path.join(CSV_FOLDER, filename))

# %%
# PLOT TIMES
times = pd.read_csv(os.path.join(CSV_FOLDER, "mean_times.csv"), index_col=0)
plot_mean_durations(FOLDER, times)

times = pd.read_csv(os.path.join(CSV_FOLDER, "std_times.csv"), index_col=0)
ax = times.plot(kind="line", title="Fitting time over missing rates")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "runtimes_deviations.png"))
plt.close(fig)
times = pd.DataFrame()

# PLOT AVG AMONG BEST K=5 FEATURES
k = 5
file_prefixes = ["mean_", "std_"]

mean_scores = pd.read_csv(
    os.path.join(CSV_FOLDER, "mean_scores.csv"), index_col=0)
std_scores = pd.read_csv(
    os.path.join(CSV_FOLDER, "std_scores.csv"), index_col=0)

for clf in classifiers:
    for p in file_prefixes:
        search_string = "{:s}{:s}*.csv".format(p, clf)
        filepaths = glob(os.path.join(CSV_FOLDER, search_string))

        for i, f in enumerate(filepaths):
            df = pd.read_csv(f, index_col=0)
            if i == 0:
                scores = pd.DataFrame(
                    np.zeros((len(missing_rates), len(df.columns) + 1)),
                    index=missing_rates,
                    columns=["complete"] + list(df.columns))

            scores.iloc[i] = df.iloc[:k].mean()

        scores["complete"] = mean_scores[clf] if "me" in p else std_scores[clf]

        t = "Average f1 {:s} among top {:d} features ({:s})".format(p, k, clf)
        ax = scores.plot(kind="line", title=t)
        ax.set(xlabel="Missing Rate", ylabel="F1")
        fig = ax.get_figure()
        filename = "{:s}f1_{:s}_{:d}.png".format(p, clf, k)
        fig.savefig(os.path.join(FOLDER, filename))
        plt.close(fig)

        if p == "mean_":
            filename = "mean_{:s}_{:d}.csv".format(clf, k)
            scores.to_csv(os.path.join(FOLDER, filename))
            aucs = calc_aucs(scores)
            plot_aucs(FOLDER, aucs, "F1", "_" + clf)

# PLOT SINGLE FILES
mr_s = [0.00]
clfs = ["knn", "tree"]
kinds = ["mean", "std"]

for clf in clfs:
    for mr in mr_s:
        title = "F1 score over features (clf={:s}, mr={:.2f})".format(clf, mr)
        for kind in kinds:
            path = os.path.join(CSV_FOLDER, "{:s}_{:s}_{:.2f}.csv".format(
                kind, clf, mr))
            df = pd.read_csv(path, index_col=0)

            if kind == "mean":
                df["complete"] = mean_scores[clf][mr]
            else:
                df["complete"] = std_scores[clf][mr]

            ax = df.plot(title=title)
            ax.set(xlabel="# Features", ylabel="F1 ({:s})".format(kind))
            fig = ax.get_figure()
            path = "{:s}_{:s}_{:.2f}.png".format(clf, kind, mr)
            fig.savefig(os.path.join(FOLDER, path))
            plt.close(fig)
