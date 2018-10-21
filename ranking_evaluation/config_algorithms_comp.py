from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.baseline import Baseline

ALGORITHMS = {
    "Baseline": {
        "class": Baseline,
        "config": {}
    },
    "XGBoost": {
        "class": Embedded,
        "config": {}
    },
    "Relief": {
        "class": Orange,
        "config": {
            "eval_method": "relief"
        }
    },
    "FCBF": {
        "class": Orange,
        "config": {
            "eval_method": "fcbf"
        }
    },
    "Random Forest": {
        "class": Orange,
        "config": {
            "eval_method": "rf"
        }
    },
    "SFS + Tree": {
        "class": SFS,
        "config": {
            "eval_method": "tree"
        }
    },
    "RKNN": {
        "class": RKNN,
        "config": {}
    },
    "MI": {
        "class": Filter,
        "config": {
            "eval_method": "mi"
        }
    },
    "CFS": {
        "class": Ranking,
        "config": {
            "eval_method": "cfs"
        }
    },
    "MRMR": {
        "class": Ranking,
        "config": {
            "eval_method": "mrmr"
        }
    },
}
