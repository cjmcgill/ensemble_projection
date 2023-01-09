import csv
import os
from typing import List, Tuple

import numpy as np
from scipy import integrate

from ensemble_projection.covariance import persistent_covariance


def get_stats(
    target_path: str,
    preds_path: str,
    ensemble_size: int,
    no_bessel_correction: bool = True,
    truncate_data_length: int = None,
    error_basis: bool = False,
    individual_preds_input: bool = False,
    covariance_calculation: bool = False,
    scratch_dir: str = None,
) -> Tuple[np.ndarray]:
    if not error_basis:
        ids, targets = load_targets(target_path)
        if individual_preds_input:
            preds_ids, preds, ensemble_vars, ind_preds = load_ind_preds(preds_path, ensemble_size, truncate_data_length)
        else:
            preds_ids, preds, ensemble_vars = load_preds(preds_path, truncate_data_length)
        if ids != preds_ids:
            raise ValueError(
                f"The datapoint identifiers in the targets from {target_path}\
                    and the datapoint identifiers in the predictions from {preds_path}\
                    must be the same."
            )
        errors = preds - targets
    else:
        if individual_preds_input:
            preds_ids, preds, ensemble_vars, ind_preds = load_ind_preds(preds_path, ensemble_size, truncate_data_length)
        else:
            preds_ids, preds, ensemble_vars = load_preds(preds_path, truncate_data_length)

    if individual_preds_input and covariance_calculation:
        covariances = np.cov(ind_preds, rowvar=True)
    else:
        covariances = None

    if not no_bessel_correction:
        ensemble_vars = bessel_correction(ensemble_vars, ensemble_size)
    covariances = bessel_correction(covariances, ensemble_size)

    if covariances is not None:
        covariance_map = persistent_covariance(covariances, scratch_dir)
    else:
        covariance_map = None

    return (
        ids,
        ensemble_vars,
        errors,
        covariance_map,
    )


def load_targets(path):
    targets = []
    ids = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            ids.append(line[0])
            targets.append(line[1])
    targets = np.array(targets, dtype=float)
    return ids, targets


def load_preds(path, truncate_data_length):
    ids = []
    preds = []
    ensemble_vars = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            ids.append(line[0])
            preds.append(line[1])
            ensemble_vars.append(line[2])
    ids = ids[:truncate_data_length]
    preds = preds[:truncate_data_length]
    ensemble_vars = ensemble_vars[:truncate_data_length]
    preds = np.array(preds, dtype=float)
    ensemble_vars = np.array(ensemble_vars, dtype=float)
    return ids, preds, ensemble_vars

def load_ind_preds(path, truncate_data_length):
    ids = []
    ind_preds = []
    ensemble_vars = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            ids.append(line[0])
            ind_preds.append(line[1:])
    ids = ids[:truncate_data_length]
    ind_preds = ind_preds[:truncate_data_length]
    ind_preds = np.array(ind_preds, dtype=float)
    preds = np.mean(ind_preds, axis=1)
    ensemble_vars = np.vars(ind_preds, axis=1)
    return ids, preds, ensemble_vars, ind_preds

def get_bw_factor(data_length: int, bw_multiplier: float, dimension: int = 1):
    bw=data_length**(-1/(dimension+4)) # Scott
    # bw=(len(s2)*(1+2)/4)**(-1/(1+4)) # Silverman 1d
    return bw*bw_multiplier


def save_iteration_stats(
    iteration: int,
    mae_iterations: List[float],
    nonvariance_iterations: List[float],
    kl_iterations: List[float],
    prior_mu: List[float],
    prior_v: List[float],
    m_mesh: List[float],
    v_mesh: List[float],
    save_dir: str,
    scratch_dir: str,
):
    stats_path = os.path.join(save_dir, "iteration_stats.csv")
    with open(stats_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "nonvariance", "expected_mae", "kl_divergence"])
        for i in range(len(mae_iterations)):
            writer.writerow([i, nonvariance_iterations[i], mae_iterations[i], kl_iterations[i]])
    priors_path = os.path.join(scratch_dir, f"prior_{iteration}.csv")
    with open(priors_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mu value", "mu prior", "v value", "v prior"])
        for i in range(max(len(m_mesh), len(v_mesh))):
            row = []
            if i < len(m_mesh):
                row.extend([m_mesh[i], prior_mu[i]])
            else:
                row.extend(["", ""])
            if i < len(v_mesh):
                row.extend([v_mesh[i], prior_v[i]])
            else:
                row.extend(["", ""])
            writer.writerow(row)


def save_iteration_stats_2d(
    iteration: int,
    mae_iterations: List[float],
    nonvariance_iterations: List[float],
    kl_iterations: List[float],
    prior_2d: List[List[float]],
    m_mesh: List[float],
    v_mesh: List[float],
    scratch_dir: str,
    save_dir: str,
):
    stats_path = os.path.join(save_dir, "iteration_stats.csv")
    with open(stats_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "nonvariance", "expected_mae", "kl_divergence"])
        for i in range(len(mae_iterations)):
            writer.writerow([i, nonvariance_iterations[i], mae_iterations[i], kl_iterations[i]])
    priors_path = os.path.join(scratch_dir, f"prior_{iteration}.csv")
    with open(priors_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mu rows / v columns"] + v_mesh.tolist())
        for i in range(len(m_mesh)):
            writer.writerow([m_mesh[i]] + prior_2d[i].tolist())


def save_projection(save_dir, projection_sizes, projected_mae, marginal_mae, nonvariance, nonvariance_std):
    projection_path = os.path.join(save_dir, "projection.csv")
    with open(projection_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ensemble size", "expected mae from new ensemble", "expected mae from marginal added models", "nonvariance error", "std confidence interval nonvariance error"])
        for i in range(len(projected_mae)):
            writer.writerow([projection_sizes[i], projected_mae[i], marginal_mae[i], nonvariance, nonvariance_std])


def bessel_correction(variances, ensemble_size):
    bessel_corrected = variances * ensemble_size / (ensemble_size - 1)
    return bessel_corrected