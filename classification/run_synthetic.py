# %%
import os
import numpy as np
import pandas as pd
from time import time
from glob import glob
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from project import EXPERIMENTS_PATH
from project.utils import Data, DataGenerator
from project.utils.imputer import Imputer
from project.utils import introduce_missing_values, scale_data
from experiments.classification.utils import get_selectors, get_classifiers
from experiments.plots import plot_mean_durations, plot_aucs
from experiments.metrics import calc_aucs

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "synthetic_half_discrete_3-1"
generator = DataGenerator(n_relevant=3, n_clusters=1, n_discrete=10)
k_s = [5]

BASE_PATH = os.path.join(EXPERIMENTS_PATH, "classification")

uses_imputation = False
if uses_imputation:
    FOLDER = os.path.join(BASE_PATH, "imputation", name)
    strategy = "knn"
    names = [
        "rar_alpha0",  # fs | transform + clf
        "rar_alpha1",  # fs | imputation + transform + clf (no cost savings)
        "rar_alpha2",  # fs | transform + imputation + clf
        "rar_alpha3",  # imputation + fs | imputation + transform + clf (no cost savings)
        "rar_alpha4",  # imputation + fs | transform + imputation + clf
        "rar_alpha5",  # imputation + fs | transform + clf
    ]
else:
    comp = [
        "baseline",
        "xgb",
        "relief",
        "fcbf",
        "rf",
        "sfs",
        "rknn",
        "mi",
        "cfs",
        "mrmr",
        "rar_deletion",
        "rar_category",
        "rar_partial",
        "rar_alpha",
        "rar_proba",
        "rar_distance",
        "rar_radius",
        "rar_multi",
    ]

    names = comp
    FOLDER = os.path.join(BASE_PATH, "incomplete", name)

CSV_FOLDER = os.path.join(FOLDER, "csv")

seeds = [17, 12, 132, 4, 7]
n_runs = 3 if len(seeds) >= 3 else len(seeds)
n_insertions = 3 if len(seeds) >= 3 else len(seeds)
classifiers = ["knn", "tree", "gnb", "svm"]
missing_rates = [0.1 * i for i in range(10)]

os.makedirs(FOLDER)
os.makedirs(CSV_FOLDER)

times = {mr: defaultdict(list) for mr in missing_rates}
complete_scores = deepcopy(times)
perfect_scores = deepcopy(times)
shuffle_seed = 0
for mr in missing_rates:
    scores_clf = {k: defaultdict(list) for k in k_s}
    scores = {algo: deepcopy(scores_clf) for algo in classifiers}

    for j in range(n_insertions):
        relevant_features, d_s = [], []
        for i_data in range(n_runs):
            generator.set_seed(seeds[i_data])
            data, relevance_vector = generator.create_dataset()
            data = scale_data(data)
            data.shuffle_rows(seed=42)
            relevant_features.append(
                relevance_vector[relevance_vector > 0].index)

            d = introduce_missing_values(data, missing_rate=mr, seed=seeds[j])
            d_s.append(d)

            if i_data == 0:
                splits = d.split()
            else:
                splits.extend(d.split())

        for i_split, (train, test) in enumerate(splits):
            i_rel = int(np.floor(i_split / n_runs))
            d = deepcopy(d_s[i_rel])
            d.shuffle_columns(seed=shuffle_seed)
            train.shuffle_columns(seed=shuffle_seed)
            test.shuffle_columns(seed=shuffle_seed)
            shuffle_seed += 1

            if uses_imputation:
                imputer = Imputer(d.f_types, strategy)
                train_imputed = imputer.complete(train)
                test_imputed = imputer.complete(test)

            # EVALUATE COMPLETE SET
            clfs = get_classifiers(train, d, classifiers)
            for i_c, clf in enumerate(clfs):
                clf.fit(train.X, train.y)
                y_pred = clf.predict(test.X)
                f1 = f1_score(test.y, y_pred, average="micro")
                complete_scores[mr][classifiers[i_c]].append(f1)

            # EVALUATE PERFECT SET
            new_X = train.X[relevant_features[i_rel]]
            new_f_types = train.f_types[relevant_features[i_rel]]
            t = Data(new_X, train.y, new_f_types, train.l_type, new_X.shape)
            clfs = get_classifiers(t, d, classifiers)
            for i_c, clf in enumerate(clfs):
                clf.fit(t.X, t.y)
                y_pred = clf.predict(test.X[relevant_features[i_rel]])
                f1 = f1_score(test.y, y_pred, average="micro")
                perfect_scores[mr][classifiers[i_c]].append(f1)

            # EVALUATE SELECTORS
            n = names
            if uses_imputation:
                n = [name[:-1] for name in names]

            previous_selector = None
            selectors = get_selectors(train, d, n, max(k_s))
            for i_s, selector in enumerate(selectors):
                start = time()

                if uses_imputation and i_s in [3, 4, 5]:
                    # run fs on imputed datasets
                    train_data = deepcopy(train_imputed)
                else:
                    train_data = deepcopy(train)

                if uses_imputation and i_s in [1, 2, 4, 5]:
                    selector = previous_selector
                else:
                    np.random.seed(seeds[i_split % n_runs])
                    selector.fit(train_data.X, train_data.y)
                    previous_selector = selector

                t = time() - start
                times[mr][names[i_s]].append(t)

                for k in k_s:
                    if uses_imputation and i_s in [1, 3]:
                        # imputation before transformation
                        X_train = selector.transform(train_imputed.X, k)
                        X_test = selector.transform(test_imputed.X, k)
                    else:
                        X_train = selector.transform(train.X, k)
                        X_test = selector.transform(test.X, k)

                    if uses_imputation and i_s in [2, 4]:
                        # impute on reduced data
                        cols = X_train.columns
                        X_train = imputer._complete(X_train, cols)
                        X_test = imputer._complete(X_test, cols)

                    f_types = train.f_types[X_train.columns]
                    transformed_data = Data(X_train, train.y, f_types,
                                            train.l_type, X_train.shape)

                    clfs = get_classifiers(transformed_data, d, classifiers)
                    for i_c, clf in enumerate(clfs):
                        clf.fit(X_train, train.y.reset_index(drop=True))
                        y_pred = clf.predict(X_test)
                        f1 = f1_score(test.y, y_pred, average="micro")
                        scores[classifiers[i_c]][k][names[i_s]].append(f1)

    for clf in classifiers:
        means = pd.DataFrame(scores[clf]).applymap(np.mean).T
        stds = pd.DataFrame(scores[clf]).applymap(np.std).T
        means.to_csv(
            os.path.join(CSV_FOLDER, "mean_{:s}_{:.2f}.csv".format(clf, mr)))
        stds.to_csv(
            os.path.join(CSV_FOLDER, "std_{:s}_{:.2f}.csv".format(clf, mr)))

# TIMES
mean_times = pd.DataFrame(times).applymap(np.mean).T
mean_times.to_csv(os.path.join(CSV_FOLDER, "mean_times.csv"))
std_times = pd.DataFrame(times).applymap(np.std).T
std_times.to_csv(os.path.join(CSV_FOLDER, "std_times.csv"))

# COMPLETE SCORES
mean_scores = pd.DataFrame(complete_scores).applymap(np.mean).T
mean_scores.to_csv(os.path.join(CSV_FOLDER, "mean_scores.csv"))
std_scores = pd.DataFrame(complete_scores).applymap(np.std).T
std_scores.to_csv(os.path.join(CSV_FOLDER, "std_scores.csv"))

# PERFECT SCORES
mean_scores = pd.DataFrame(perfect_scores).applymap(np.mean).T
mean_scores.to_csv(os.path.join(CSV_FOLDER, "mean_perfect_scores.csv"))
std_scores = pd.DataFrame(perfect_scores).applymap(np.std).T
std_scores.to_csv(os.path.join(CSV_FOLDER, "std_perfect_scores.csv"))

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
k = 1
file_prefixes = ["mean_", "std_"]

mean_scores = pd.read_csv(
    os.path.join(CSV_FOLDER, "mean_scores.csv"), index_col=0)
std_scores = pd.read_csv(
    os.path.join(CSV_FOLDER, "std_scores.csv"), index_col=0)
mean_perfect_scores = pd.read_csv(
    os.path.join(CSV_FOLDER, "mean_perfect_scores.csv"), index_col=0)
std_perfect_scores = pd.read_csv(
    os.path.join(CSV_FOLDER, "std_perfect_scores.csv"), index_col=0)

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
        scores["perfect"] = mean_perfect_scores[
            clf] if "me" in p else std_perfect_scores[clf]

        t = "Average f1 ({:s})".format(clf)
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
