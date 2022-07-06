from typing import List, Literal

from tap import Tap


class Args(Tap):
    """:class:`Args` contains arguments that are used in quantifying and projecting variance error."""

    target_path: str
    """target file containing one column with datapoint identifier and one column with targets"""
    preds_path: str
    """preds file containing columns with datapoint identifier, predicted value, and ensemble variance"""
    ensemble_size: int
    """the number of ensemble models used in the predictions in the preds file"""
    save_dir: str
    """the directory where you want to save the projection results"""

    convergence_method: Literal["iteration_count"] = "iteration_count"
    optimization_iterations: int = 10
    """The number of iterations to use in refining the prior distribution"""
    bessel_correction_needed: bool = True
    """Whether to apply Bessel's correction to sample variances in the preds file."""
    truncate_data_length: int = None
    """The number of data to use in projection. Defaults to using all data."""
    bw_multiplier: float = 1
    """A multiplier for the width used in gaussian kernel density estimation for the initial prior distribution."""
    mu_mesh_size: int = 1025
    """The number of evaluation points to use in integration, across the mean axis."""
    v_mesh_size: int = 129
    """The number of evaluation points to use in integration, across the variance axis."""
    prior_method: Literal["separate", "combined"] = "combined"
    """What function form to use for the prior distribution."""
    max_projection_size: int = 100
    """The largest ensemble size to project out performance to."""
    save_iteration_steps: bool = False
    """Wheter to save the incremental stats for each iteration. Increases the calculation time. Useful for debug."""