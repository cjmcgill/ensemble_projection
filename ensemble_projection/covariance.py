import os
import numpy as np

def persistent_covariance(covariances, scratch_dir):
    """Returns a memmap array for the covariance matrix of shape(data, data)."""
    covariance_path = os.path.join(scratch_dir, "covariance.dat")
    covariance_map = np.memmap(
        filename=covariance_path,
        dtype=float,
        mode="w+",
        shape=(len(covariances))
    )
    covariance_map[:] = covariances
    del likelihood_map
    likelihood_map = np.memmap(
        filename=covariance_path,
        dtype=float,
        mode="r",
        shape=(len(covariances))
    )
    return covariance_map


def remove_covariance(scratch_dir):
    covariance_path = os.path.join(scratch_dir, "covariance.dat")
    os.remove(covariance_path)
