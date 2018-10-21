# %%
import os
import pandas as pd
from project import EXPERIMENTS_PATH
from glob import glob

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
clf = "sk_knn"
clf = "gnb"
imp = "knn"
names = [
    "heart-c", "hepatitis", "ionosphere", "isolet", "madelon", "musk",
    "schizo", "semeion", "soybean", "vote"
]
names = ["heart-c", "hepatitis", "ionosphere", "schizo", "soybean", "vote"]
path = os.path.join(EXPERIMENTS_PATH, "classification", "imputation")

scores = {name: 0 for name in names}
for i, name in enumerate(names):
    pa = os.path.join(path, name + "_" + imp + "_del_nmar")
    c = clf
    if name in ["isolet", "semeion", "musk", "madelon"] and clf == "knn":
        c = "sk_knn"

    paths = glob(os.path.join(pa, "csv", "mean_" + c + "*.csv"))[:5]
    for p in paths:
        mr = float(p.split("_")[-1].split(".csv")[0])
        df = pd.read_csv(p, index_col=0).mean() * 1 / 5
        scores[name] += df

    p = os.path.join(pa, "csv", "mean_scores.csv")
    df = pd.read_csv(p, index_col=0).mean()
    scores[name]["complete"] = df[c]

scores = pd.DataFrame(scores)
scores.to_csv(os.path.join(path, "aucs_mean" + clf + imp + "_nmar" + ".csv"))
scores
