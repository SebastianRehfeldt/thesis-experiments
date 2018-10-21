# %%
import os
import pandas as pd
from project import EXPERIMENTS_PATH
from glob import glob

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
clf = "knn"
imp = "knn"
noise = ""
noise = "_noise"
names = [
    "heart-c", "hepatitis", "ionosphere", "isolet", "madelon", "musk",
    "schizo", "semeion", "soybean", "vote"
]
names = ["heart-c", "hepatitis", "ionosphere", "schizo", "soybean", "vote"]
path = os.path.join(EXPERIMENTS_PATH, "classification", "imputation")

scores = {name: 0 for name in names}
for i, name in enumerate(names):
    pa = os.path.join(path, name + "_" + imp + noise)
    c = clf
    if name in ["isolet", "semeion", "musk", "madelon"] and clf == "knn":
        c = "sk_knn"

    paths = glob(os.path.join(pa, "csv", "mean_" + c + "*.csv"))[:8]
    for p in paths:
        mr = float(p.split("_")[-1].split(".csv")[0])
        df = pd.read_csv(p, index_col=0).max() * 1 / 8
        scores[name] += df

    p = os.path.join(pa, "csv", "mean_scores.csv")
    df = pd.read_csv(p, index_col=0).mean()
    scores[name]["complete"] = df[c]

scores = pd.DataFrame(scores)
# scores.to_csv(os.path.join(path, "aucs_max_" + clf + imp + noise + ".csv"))

for val in scores.sum(1).values:
    print(str(val).replace(".", ","))
