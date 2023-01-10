from typing import List, Literal

from tap import Tap


class Args(Tap):
    """:class:`Args` contains arguments that are used in quantifying and projecting variance error."""

    target_path: str = None
    """target file containing one column with datapoint identifier and one column with targets"""
    preds_path: str
    """preds file containing columns with datapoint identifier, predicted value, and ensemble variance"""
    ensemble_size: int
    """the number of ensemble models used in the predictions in the preds file"""
    save_dir: str
    """the directory where you want to save the projection results"""
    scratch_dir: str = None
    """location of temporary directory to store temporary files"""
    individual_preds_input: bool = False
    """"""

    # Convergence Settings
    convergence_method: Literal["iteration_count", "kl_threshold", "fraction_change_threshold"] = "iteration_count"
    """What method to use when determining whether the iterative calculation has converged."""
    kl_threshold: float = 1e-5
    """Sets the kl divergence criterion for terminating iteration"""
    fraction_change_threshold: float = 1e-4
    """Sets the fractional change of mean nonvariance error criterion for terminating iteration"""
    optimization_iterations: int = 10
    """The number of iterations to use in refining the prior distribution"""

    prior_method: Literal["separate", "combined"] = "combined"
    """What function form to use for the prior distribution."""
    initial_prior: Literal["kde", "likelihood", "gaussian", "uniform"] = "kde"
    """What to use for the initial prior distribution. Likelihood only set up for combined prior."""
    no_bessel_correction_needed: bool = False
    """Whether to apply Bessel's correction to sample variances in the preds file."""
    truncate_data_length: int = None
    """The number of data to use in projection. Defaults to using all data."""
    bw_multiplier: float = 1
    """A multiplier for the width used in gaussian kernel density estimation for the initial prior distribution."""
    mu_mesh_size: int = 1025
    """The number of evaluation points to use in integration, across the mean axis."""
    v_mesh_size: int = 129
    """The number of evaluation points to use in integration, across the variance axis."""
    max_projection_size: int = 100
    """The largest ensemble size to project out performance to."""
    save_iteration_steps: bool = False
    """Wheter to save the incremental stats for each iteration. Increases the calculation time. Useful for debug."""
    learning_rate: float = 1.0
    """An exponential factor to adjust the step size taken in each prior update."""
    error_basis: bool = False
    """Whether the predictions provided are already expressed as errors, i.e. targets are all zero."""
    integration_method: Literal["trapz", "simps"] = "trapz"
    """What function to use for integrating across the distribution"""
    likelihood_calculation: Literal["persistent", "calculated"] = "persistent"
    """Sets whether the likelihood distribution is recalculated each iteration (calc time) or calculated once and saved (disk)"""