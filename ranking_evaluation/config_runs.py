CONFIG = {
    "n_runs":
    5,
    "n_insertions":
    5,  # maximum 10 insertions
    "seeds": [42, 0, 13, 84, 107, 15, 23, 11, 174, 147],
    "missing_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # "missing_rates": [0.0],
    "is_real_data":
    False,
    "update_config":
    True,  # update params must be >= n_runs
    "update_attribute":
    "n_features",
    # "updates": [
    #     {
    #         "n_samples": 250,
    #     },
    #     {
    #         "n_samples": 500,
    #     },
    #     {
    #         "n_samples": 1000,
    #     },
    #     {
    #         "n_samples": 1500,
    #     },
    #     {
    #         "n_samples": 2000,
    #     },
    # ],
    "updates": [
        {
            "n_features": 20,
            "n_independent": 20,
            "n_relevant": 3,
            "n_discrete": 10,
            "n_clusters": 1,
        },
        {
            "n_features": 40,
            "n_independent": 40,
            "n_relevant": 6,
            "n_discrete": 20,
            "n_clusters": 2,
        },
        {
            "n_features": 60,
            "n_independent": 60,
            "n_relevant": 9,
            "n_discrete": 30,
            "n_clusters": 3,
        },
        {
            "n_features": 80,
            "n_independent": 80,
            "n_relevant": 12,
            "n_discrete": 40,
            "n_clusters": 4,
        },
        {
            "n_features": 100,
            "n_independent": 100,
            "n_relevant": 15,
            "n_discrete": 50,
            "n_clusters": 5,
        },
    ],
}
