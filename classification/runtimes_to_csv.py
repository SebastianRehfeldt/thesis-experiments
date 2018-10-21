# %%
import os
import pandas as pd
from project import EXPERIMENTS_PATH
from glob import glob

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
clf = "knn"
imp = "knn"
names = ["heart-c", "hepatitis", "ionosphere", "schizo", "soybean", "vote"]
path = os.path.join(EXPERIMENTS_PATH, "classification", "imputation")

for i, name in enumerate(names):
    p = os.path.join(path, name + "_" + imp + "_noise")
    c = clf
    if name in ["isolet", "semeion", "musk", "madelon"] and clf == "knn":
        c = "sk_knn"
    pa = os.path.join(p, "AUC", "aucs_{:s}.csv".format(c))
    if i == 0:
        index = pd.read_csv(pa, index_col=0, header=None).index
        scores = pd.DataFrame(index=index)
    scores[name] = pd.read_csv(pa, index_col=0, header=None)

scores.to_csv(os.path.join(path, "aucs" + clf + imp + "noise" + ".csv"))
