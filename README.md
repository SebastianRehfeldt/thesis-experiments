# Experimental Results for Feature Selection on Incomplete Datasets

This repository contains scripts to run the experiments presented in my Master's thesis. It further includes the most important results and scripts to combine and aggregate results of different experiments.

## Aggregated Results

Aggregated data, which are also presented in the thesis, can be found in the results_uci and results_synthetic folders in form of plots and Excel files. The excel files contain separated sheets for each experimental section of the thesis.

## Raw Results

There are two main folders storing the raw results: classification and ranking_evaluation.

### Classification

There are two subfolders. In the **imputation** folder the impact of imputation on feature selection is assessed. The dataset name is followed by the imputation technique and the missing mechanism (default is MCAR). The **incomplete** folder contains evaluations of classification directly on incomplete datasets. The suffixes describe again the missing mechanisms or represent the feature selection approaches.

### Ranking Evaluation

In this folder ranking metrics are used to compare feature selection approaches. In the **synthetic** folder you can find experiments on artificial data and in **uci** on real-world data. The **updates** folder contains experiments on artificial data with updating configurations to assess scalability.
Each folder contains a configuration files which includes the dataset parameters, a configuration of runs and a list of feature selection approaches with all important parameters.