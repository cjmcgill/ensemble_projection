from typing import Callable
from abc import ABC, abstractmethod

import numpy as np


class ConvergenceCriterion(ABC):
    """Class of convergence checkers to see if the optimization is converged."""
    def __init__(
        self,
        optimization_iterations: int,
        kl_threshold: float,
        fraction_change_threshold: float,
    ):
        self.optimization_iterations = optimization_iterations
        self.kl_threshold = kl_threshold
        self.fraction_change_threshold = fraction_change_threshold
    
    @abstractmethod
    def is_not_converged(
        self,
        iteration: int,
        previous_nonvariance: np.ndarray,
        current_nonvariance: np.ndarray,
        kl: float,
    ) -> bool:
        """Checks whether the method has reached convergence"""


class IterationCriterion(ConvergenceCriterion):
    def is_not_converged(self, iteration: int, previous_nonvariance: np.ndarray, current_nonvariance: np.ndarray, kl: float) -> bool:
        return iteration < self.optimization_iterations


class KLCriterion(ConvergenceCriterion):
    def is_not_converged(self, iteration: int, previous_nonvariance: np.ndarray, current_nonvariance: np.ndarray, kl: float) -> bool:
        if kl is None:
            return True
        else:
            return kl > self.kl_threshold


class ChangeCriterion(ConvergenceCriterion):
    def is_not_converged(self, iteration: int, previous_nonvariance: np.ndarray, current_nonvariance: np.ndarray, kl: float) -> bool:
        if previous_nonvariance is None:
            return True
        else:
            return np.abs(current_nonvariance - previous_nonvariance) / previous_nonvariance > self.fraction_change_threshold



def get_convergence_checker(
    convergence_method: str,
    optimization_iterations: int,
    kl_threshold: float,
    fraction_change_threshold: float,
) -> ConvergenceCriterion:

    supported_convergence_methods = {
        "iteration_count": IterationCriterion,
        "fraction_change_threshold": ChangeCriterion,
        "kl_threshold": KLCriterion,
    }
    convergence_class = supported_convergence_methods.get(convergence_method)
    if convergence_class is None:
        raise NotImplementedError(
            f"Convergence method {convergence_method} is not supported"
        )

    return convergence_class(optimization_iterations, kl_threshold, fraction_change_threshold)
