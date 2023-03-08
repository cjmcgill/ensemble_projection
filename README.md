# Ensemble Projection
Calculate the expected error for a dataset using different sizes of model ensembles, given the true values of the dataset and the predictions of an ensemble model of known size. This method is used after model training and is model-type agnostic; it can be used with any regression model architecture.

This method is used in the paper "Characterizing Uncertainty in Machine Learning for Chemistry" (https://doi.org/10.26434/chemrxiv-2023-00vcg) and a forthcoming manuscript in progress focusing on this method specifically.

## Requirements

The dependencies necessary to use the ensemble projection program are included in the base anaconda environment. Anaconda can be installed from [https://www.anaconda.com/](https://www.anaconda.com/). The minimum requirements can be found in the `environment.yml` file.

To install using the minimum necessary environment:

1. `git clone https://github.com/chemprop/chemprop.git`
2. `cd ensemble_projection`
3. `conda env create -f environment.yml`

## Using Ensemble Projection

Ensemble projection can be used with the following command line prompt.

`python {path-to-ensemble-projection}/main.py --target_path {path-to-targets-file.csv} --preds_path {path-to-predictions-file.csv} --ensemble_size {n} --save_dir {dir-to-save-results}`

These are the minimum necessary inputs for performing ensemble projection. Descriptions of these and other arguments can be found in the `args.py` file.

## Data Format

The target and predictions files must be formatted as a **CSV file with a header row**.

The targets file must contain two columns. The first column must contain unique data identifiers. The second column contains the regression target.

```
ID, regression_property
001, 0.572
002, 0.639
...
```

The predictions file in its base form contains three columns. The first column contains the unique data identifiers, which must match those of the target file. The second column contains the property prediction. The third column contains the variance of predictions made by the ensemble of models.

```
ID, regression_prediction, ensemble_variance
001, 0.54, 0.24
002, 0.7, 0.73
...
```

## Results Output

The main results file for ensemble projection is called `projection.csv`. It contains four outputs.
* Expected MAE from a new ensemble. This indicates the expected value for the MAE of this dataset for different ensemble sizes when the new ensemble is trained from scratch.
* Expected MAE from marginal added models. This indicates the expected value for the MAE of this dataset for different ensemble sizes when added to the existing models used in the initial prediction ensemble.
* Nonvariance error. This provides the expected value for the component of the MAE that is not due to variance error (either from model bias or data noise error), as calculated using the posterior distribution of prediction values. The nonvariance error component is not a function of ensemble size. It can be interpreted as the expected MAE for an infinitely large ensemble.
* Confidence interval. This provides the standard deviation of the calculated nonvariance error, as calculated using the posterior distribution of prediction values. It can be interpreted as the spread of possible realized nonvariance errors around the expected value.

## Convergence Settings

The ensemble projection method uses an iterative refinement of Bayesian inference. The calculation is completed when it is judged to be converged.

The method of determining convergence is chosen with the `--convergence_method` argument, with the following options.
* `iteration_count` (default). Stops after a predetermined number of iterations. The number of iterations can be chosen using the `--optimization_iterations {int,default=10}` argument.
* `kl_threshold`. Considers the calculation converged when the Kullback-Leibler divergence between the prior distribution in one iteration and the next is less than a threshold value. The threshold can be changed using `--kl_threshold {float}`.
* `fraction_change_threshold`. Considers the calculation converged when the expected nonvariance error changes by a fractional amount below the specified threshold between iterations. Can change this threshold with `--fraction_change_threshold {float}`.
