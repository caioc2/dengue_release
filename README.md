# "Predicting dengue outbreaks in Brazil with manifoldlearning on climate data"

Caio Souza, Pedro Maia, Lucas M. Stolerman, Vitor Rolla and Luiz Velho

## Description

In this work, we improve upon a recent approach of coarsely predicting outbreaks in Brazilian urban centers based solely on their yearly climate data. Our methodological advancements encompass a judicious choice of data pre-processing steps and usage of modern computational techniques from signal-processing and manifold  learning.

## Organization

***/data/*** folder contains the climate and dengue outbreak data for the cities of Aracaju, Belo Horizonte, Manaus, Recife, Rio de Janeiro, Salvador and São Luís. Each sub-folder contains the files: *dengue.csv* (dengue cases and incidence per year), *precip.csv* (daily measure for precipitation), *temp_avg.csv* (daily measure for the average temperature) and *years.csv* (correspondent year for each line in the previous files).

***/code/run_grid.py*** is resposible for running the grid search over the hyper-parameters and selecting the best prediction date and model. The final result is writen to */results/result.csv*, while intermediate results for the grid and selection, including plots for each model, are in */results/intermediate/*.

***/code/plot_region.py*** is resposible for generating the main figures with the classifier regions for the previous found best hyper-parameters and date (*/results/figures/[city]/*).

***/code/calc_stats.py*** is resposible for calculating the statistic tests for the models and writes to */results/stats.csv*.

### Important notes

Our datasets are small, about 14-16 years for each city, given that, we use noisy data for validation (tuning the hyper parameters). For that reason the results may have slightly variations from run to run. For the grid search step, a complete list of the sorted grid for each city can be found at */results/intermediate/selection/[city]/*.

The same aforementioned reason may interfere with the Student's Ttest and McNemar Test for the random guess dummy classifier, as it generates a small amout of samples, the variability may be higher than for large datasets, where the samples should be 50/50 for the positve and negative class, for this reason we also include the same tests for the moda classifier, which shoud be constant for a given model.

## Environment and Dependencies

These scripts depends on the python libraries: *scikit-learn, pandas, matplotlib* and *statsmodels* and were tested on both Windows and Linux systems. The grid search may take approximated 1 hour to run while the other scripts should take just a few seconds. The times were measured in a standard notebook with a i7 7700HQ and 16GB RAM.