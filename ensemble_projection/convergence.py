from typing import Callable
from abc import ABC, abstractmethod

import numpy as np


class ConvergenceCriterion(ABC):
    """Class of convergence checkers to see if the optimization is converged."""
    def __init__(
        self,
        optimization_iterations: int,
    ):
        self.optimization_iterations = optimization_iterations
    
    @abstractmethod
    def is_not_converged(
        self,
        iteration: int,
        previous_distribution: np.ndarray,
        current_distribution: np.ndarray,
    ) -> bool:
        """Checks whether the method has reached convergence"""


class IterationCriterion(ConvergenceCriterion):
    def is_not_converged(self, iteration: int, previous_distribution: np.ndarray, current_distribution: np.ndarray):
        return iteration < self.optimization_iterations


def get_convergence_checker(
    convergence_method: str,
    optimization_iterations: int,
) -> ConvergenceCriterion:

    supported_convergence_methods = {
        "iteration_count": IterationCriterion,
    }
    convergence_class = supported_convergence_methods.get(convergence_method)
    if convergence_class is None:
        raise NotImplementedError(
            f"Convergence method {convergence_method} is not supported"
        )

    return convergence_class(optimization_iterations)
