from copy import deepcopy

from project.rar.rar import RaR
from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.baseline import Baseline
from project.classifier import KNN, Tree, SKClassifier


def get_selectors(data, complete, names, max_k=None):
    d = [data.f_types, data.l_type, data.shape]
    d2 = [complete.f_types, complete.l_type, complete.shape]
    max_k = data.shape[1] if max_k is None else max_k

    selectors = {
        "rar_deletion":
        RaR(*d, approach="deletion"),
        "rar_category":
        RaR(*d, approach="deletion", create_category=True),
        "rar_partial":
        RaR(*d, approach="partial"),
        "rar_alpha":
        RaR(*d, approach="fuzzy", weight_approach="alpha"),
        "rar_proba":
        RaR(*d, approach="fuzzy", weight_approach="proba"),
        "rar_distance":
        RaR(*d,
            approach="fuzzy",
            weight_approach="imputed",
            dist_method="distance",
            imputation_method="soft"),
        "rar_radius":
        RaR(*d,
            approach="fuzzy",
            weight_approach="imputed",
            dist_method="radius",
            imputation_method="soft"),
        "rar_multi":
        RaR(*d, approach="fuzzy", weight_approach="multiple"),
        "baseline":
        Baseline(*d),
        "xgb":
        Embedded(*d),
        "relief":
        Orange(*d, eval_method="relief"),
        "fcbf":
        Orange(*d, eval_method="fcbf"),
        "rf":
        Orange(*d, eval_method="rf"),
        "sfs":
        SFS(*d2, k=max_k, do_stop=True, eval_method="tree"),
        "rknn":
        RKNN(*d),
        "mi":
        Filter(*d),
        "cfs":
        Ranking(*d, eval_method="cfs"),
        "mrmr":
        Ranking(*d, eval_method="mrmr"),
        "rar_alpha-as-50":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=True,
            n_subspaces=50),
        "rar_alpha-as-145":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=True,
            n_subspaces=145),
        "rar_alpha-as-290":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=True,
            n_subspaces=290),
        "rar_alpha-50":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=False,
            n_subspaces=50),
        "rar_alpha-145":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=False,
            n_subspaces=145),
        "rar_alpha-290":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=False,
            n_subspaces=290),
        "rar_alpha-580":
        RaR(*d,
            approach="fuzzy",
            weight_approach="alpha",
            active_sampling=False,
            n_subspaces=580),
        "rar_alpha-b0":
        RaR(*d, approach="fuzzy", weight_approach="alpha", boost_value=0),
        "rar_alpha-b0.1":
        RaR(*d, approach="fuzzy", weight_approach="alpha", boost_value=0.1),
        "rar_alpha-b0.2":
        RaR(*d, approach="fuzzy", weight_approach="alpha", boost_value=0.2),
        "rar_alpha-c0":
        RaR(*d, approach="fuzzy", weight_approach="alpha", boost_corr=0),
        "rar_alpha-c0.1":
        RaR(*d, approach="fuzzy", weight_approach="alpha", boost_corr=0.1),
        "rar_alpha-c0.2":
        RaR(*d, approach="fuzzy", weight_approach="alpha", boost_corr=0.2),
    }

    return [deepcopy(selectors[name]) for name in names]


def get_classifiers(data, complete, names):
    classifiers = {
        "knn": KNN(data.f_types, data.l_type, knn_neighbors=6),
        "tree": Tree(complete.to_table().domain),
        "svm": SKClassifier(data.f_types, "svm"),
        "gnb": SKClassifier(data.f_types, "gnb"),
        "xgb": SKClassifier(data.f_types, "xgb"),
        "log": SKClassifier(data.f_types, "log"),
        "sk_knn": SKClassifier(data.f_types, "knn"),
        "sk_tree": SKClassifier(data.f_types, "tree"),
    }
    return [classifiers[name] for name in names]
