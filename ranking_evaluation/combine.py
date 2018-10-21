# %%
import os
import pandas as pd
from glob import glob
from shutil import copyfile
from project import EXPERIMENTS_PATH
from experiments.ranking import calc_mean_ranking
from experiments.utils import write_results, read_results

exp = "_heart-c_"
BASE_PATH = os.path.join(EXPERIMENTS_PATH, "ranking_evaluation", "uci")
is_real_data = "uci" in BASE_PATH
PATHS = glob(os.path.join(BASE_PATH, "*" + exp + "*"))

FOLDER = os.path.join(BASE_PATH, "EXP" + exp + "combined")
os.makedirs(FOLDER)

for i, path in enumerate(PATHS):
    if "combined" in path:
        continue

    # CONFIGS
    src = os.path.join(path, "config.txt")
    dst = os.path.join(FOLDER, "config.txt")
    if i == 0:
        copyfile(src, dst)
    else:
        with open(src, 'r') as f:
            read_data = f.read().split("CONFIG ALGORITHMS")[-1]
        with open(dst, 'a') as f:
            f.write(read_data)

    # RELEVANCES
    if i == 0 and not is_real_data:
        src = os.path.join(path, "relevances.csv")
        dst = os.path.join(FOLDER, "relevances.csv")
        copyfile(src, dst)
        relevances = pd.read_csv(src, index_col=0)

    # RANKINGS, DURATIONS
    if i == 0:
        relevances, durations, rankings = read_results(path)
    else:
        _, durations_new, rankings_new = read_results(path)
        for key in durations.keys():
            durations[key].update(durations_new[key])
            rankings[key].update(rankings_new[key])

write_results(FOLDER, relevances, durations, rankings)

# CALC ADDITIONAL STATISTICS (MEAN_DURCATIONS; CG; NDCG; SSE; MSE)
from experiments.utils import get_mean_durations
from experiments.metrics import compute_statistics, calc_aucs

mean_scores = calc_mean_ranking(rankings)
mean_durations = get_mean_durations(durations)
if is_real_data:
    statistics = compute_statistics(rankings, mean_scores, mean_scores)
    cgs, cgs_at, ndcgs, cgs_pos, ndcgs_pos, sses, mses = statistics
    aucs = calc_aucs(ndcgs_pos[0])
else:
    statistics = compute_statistics(rankings, relevances, mean_scores)
    cgs, cgs_at, ndcgs, cgs_pos, ndcgs_pos, sses, mses = statistics
    aucs = calc_aucs(cgs_at[0])

# PLOT RESULTS
from experiments.plots import plot_mean_durations, plot_cgs, plot_scores
from experiments.plots import plot_aucs

plot_mean_durations(FOLDER, mean_durations)
plot_aucs(FOLDER, aucs)
plot_scores(FOLDER, cgs_at, "CG_AT_N")
plot_scores(FOLDER, ndcgs, "NDCG")
plot_scores(FOLDER, ndcgs_pos, "NDCG_POS")
plot_scores(FOLDER, sses, "SSE")
plot_scores(FOLDER, mses, "MSE")
plot_cgs(FOLDER, cgs, "CG")
plot_cgs(FOLDER, cgs_pos, "CG_POS")
