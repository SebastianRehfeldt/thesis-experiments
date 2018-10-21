# %%
import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from project import EXPERIMENTS_PATH
from project.rar.rar import RaR
from project.classifier.sklearn_classifier import SKClassifier
from project.utils import DataLoader, scale_data
from project.utils.data import DataGenerator
from experiments.metrics import calc_cg

# CHANGES FOR EXPERIMENT
is_real_data = True
change_alpha = False
change_n_subspaces = False

samples_per_class = [10]
if is_real_data:
    name = "anneal"
    k_thresh = 15
    samples_per_class = [5]

if change_n_subspaces:
    min_subspaces = 50
    max_subspaces = 1200

# DEFAULT PARAMS
alpha = [None]
subspace_sizes = [None]
active_sampling = [False]
k = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

# LOAD DATA AND SELECT K BASED ON THRESH
if is_real_data:
    k = [x for x in k if x <= k_thresh]
    data_loader = DataLoader(ignored_attributes=["molecule_name"])
    data = data_loader.load_data(name, "arff")
else:
    dataset_params = {
        "n_relevant": 3,
        "n_clusters": 3,
        "n_discrete": 10,
    }
    n_subspaces = [1000]
    generator = DataGenerator(13, **dataset_params)
    data, relevances = generator.create_dataset()
    name = "_".join(map(str, list(generator.get_params().values())))
    n_relevant = int((relevances > 0).sum())
    k = [x for x in k if x <= n_relevant]

# PARAM UPDATES
if change_alpha:
    alpha = [0.2, 0.1, 0.05, 0.02, 0.01, None]
    subspace_sizes = [(1, 3), (2, 3)]

if change_n_subspaces:
    active_sampling = [True, False]
    n_subspaces = [50, 100, 200, 400, 600, 900, 1200, 1600, 2000]
    n_subspaces = [
        n for n in n_subspaces if min_subspaces <= n <= max_subspaces
    ] + [None]

# PREPARE DATA AND FOLDER
from project.utils.data_modifier import introduce_missing_values
data = scale_data(data)
data = introduce_missing_values(data, 0)
data.shuffle_rows(seed=42)

FOLDER = os.path.join(EXPERIMENTS_PATH, "grid", "3_fixed_alpha", name)
os.makedirs(FOLDER)

# INIT RAR AND KNN
knn = SKClassifier(data.f_types, kind="knn")
rar = RaR(data.f_types, data.l_type, data.shape)
pipe = Pipeline([('reduce_dim', rar), ('classify', knn)])

# PARAM GRID
param_grid = [
    {
        'reduce_dim': [RaR(data.f_types, data.l_type, data.shape)],
        'reduce_dim__k': k,
        'reduce_dim__approach': ["deletion"],
        'reduce_dim__alpha': alpha,
        'reduce_dim__n_subspaces': [None],
        'reduce_dim__subspace_size': subspace_sizes,
        "reduce_dim__active_sampling": active_sampling,
        "reduce_dim__samples_per_class": samples_per_class,
        'reduce_dim__seed': [42],
        'reduce_dim__boost_value': [0],
        "reduce_dim__boost_inter": [0],
        "reduce_dim__boost_corr": [0],
    },
]

# FIT GRID SEARCH
grid = GridSearchCV(
    pipe,
    cv=3,
    n_jobs=4,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, average="micro"),
)
grid.fit(data.X, data.y)

# STORE RESULTS AND RANKING
results = pd.DataFrame(grid.cv_results_)
results.sort_values(by="mean_test_score", ascending=False, inplace=True)
results.to_csv(os.path.join(FOLDER, "results.csv"))

ranking = grid.best_estimator_.steps[0][1].get_ranking()
ranking = dict(ranking)
ranking = pd.DataFrame(
    data={"score": list(ranking.values())},
    index=ranking.keys(),
)
ranking.to_csv(os.path.join(FOLDER, "ranking.csv"))

# STORE BEST PARAMS
with open(os.path.join(FOLDER, "config.json"), 'w') as filepath:
    best_params = results.iloc[0].params["reduce_dim"].get_params()
    del best_params["f_types"]
    del best_params["l_type"]
    del best_params["shape"]
    json.dump(best_params, filepath, indent=4)

# STORE PARAMS GRID
with open(os.path.join(FOLDER, "param_grid.json"), 'w') as filepath:
    params = param_grid.copy()[0]
    del params["reduce_dim"]
    json.dump(param_grid, filepath, indent=4)

if not is_real_data:
    relevances.sort_values(ascending=False, inplace=True)
    relevances.to_csv(os.path.join(FOLDER, "relevances.csv"))
    with open(os.path.join(FOLDER, "data_config.json"), 'w') as filepath:
        json.dump(generator.get_params(), filepath, indent=4)
    print(calc_cg(relevances, ranking.to_dict()['score'])[n_relevant])
