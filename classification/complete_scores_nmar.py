# %%
import os
import pandas as pd
from project import EXPERIMENTS_PATH
from glob import glob

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
classifiers = ["gnb", "knn", "tree", "svm"]
FOLDER = os.path.join(EXPERIMENTS_PATH, "classification", "incomplete")
paths = glob(os.path.join(FOLDER, "*_nmar*"))

scores = {clf: {} for clf in classifiers}
for path in paths:
    name = path.split("\\")[-1]
    for clf in classifiers:
        p = os.path.join(path, "AUC", "aucs_" + clf + ".csv")
        s = pd.read_csv(p, index_col=0, header=None)
        scores[clf][name] = float(s.loc["complete"])

    name = name.split("_")[0]
    for clf in classifiers:
        p = os.path.join(path, "AUC", "aucs_" + clf + ".csv")
        p = p.replace("_nmar", "_comp")
        s = pd.read_csv(p, index_col=0, header=None)
        scores[clf][name] = float(s.loc["complete"])

p = os.path.join(FOLDER, "nmar_complete.csv")
pd.DataFrame(scores).to_csv(p)
