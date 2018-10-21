# %%
import os
import numpy as np
import pandas as pd
from glob import glob
from project import EXPERIMENTS_PATH
from experiments.utils import read_results

data = "synthetic"
exp = "EXP_3-1_rar_imp"
BASE_PATH = os.path.join(EXPERIMENTS_PATH, "ranking_evaluation", data)
is_real_data = "uci" in BASE_PATH
PATHS = glob(os.path.join(BASE_PATH, exp))

for i, path in enumerate(PATHS):
    relevances, durations, rankings = read_results(path)
    df = pd.DataFrame(durations)
    df = df.applymap(np.mean).T
    if data == "uci":
        df = df.max()

    df.to_csv(os.path.join(path, "runtimes.csv"))
