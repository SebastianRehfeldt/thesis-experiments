from project.rar.rar import RaR

SHARED = {}

ALGORITHMS = {
    "RaR-deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "RaR-category": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "create_category": True,
            **SHARED
        }
    },
    "RaR-partial": {
        "class": RaR,
        "config": {
            "approach": "partial",
            **SHARED
        }
    },
    "RaR-alpha": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "alpha",
            **SHARED
        }
    },
    "RaR-proba": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "proba",
            **SHARED
        }
    },
    "RaR-distance": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "imputation_method": "mice",
            "dist_method": "distance",
            **SHARED
        }
    },
    "RaR-radius": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "imputation_method": "mice",
            "dist_method": "radius",
            **SHARED
        }
    },
    "RaR-multi": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "multiple",
            **SHARED
        }
    },
    "MEAN + RaR": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "DELETION + RaR": {
        "should_delete": True,
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
}
