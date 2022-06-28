import os
from typing import Tuple

import numpy as np
from scipy import integrate

from ensemble_projection.utils import get_stats, get_bw_factor, kl_divergence, save_iteration_stats, save_projection
from ensemble_projection.convergence import get_convergence_checker
from ensemble_projection.kde import y_kde, s2_kde
from ensemble_projection.distribution_calculations import calc_denominator, expected_marginal_mae, expected_nonvariance, expected_mae, update_separate_priors


def bayes_infer(
    target_path: str,
    preds_path: str,
    ensemble_size: int,
    save_dir: str,
    convergence_method: str = "iteration_count",
    optimization_iterations: int = 10,
    bessel_correction_needed: bool = True,
    truncate_data_length: int = None,
    bw_multiplier: float = 1.,
    integration_mesh_size: int = 1025,
):
    """
    Perform a bayesian inference calculations for the error of a NN
    prediction without variance error included. Use this information
    to quantify the variance and nonvariance error and project the
    total error with different numbers of ensemble models used.
    """

    ids, targets, preds, ensemble_vars, errors = get_stats(
        target_path=target_path,
        preds_path=preds_path,
        ensemble_size=ensemble_size,
        truncate_data_length=truncate_data_length,
        bessel_correction=bessel_correction_needed,
    )
    os.makedirs(os.path.join(save_dir, "scratch"), exist_ok=True)
    separate_priors(
        y=errors,
        s2=ensemble_vars,
        ensemble_size=ensemble_size,
        convergence_method=convergence_method,
        optimization_iterations=optimization_iterations,
        save_dir=save_dir,
        bw_multiplier=bw_multiplier,
        integration_mesh_size=integration_mesh_size,
    )


def separate_priors(
    y: np.ndarray,
    s2: np.ndarray,
    ensemble_size: int,
    convergence_method: str,
    optimization_iterations: int,
    save_dir: str,
    bw_multiplier: float = 1.,
    integration_mesh_size: int = 1025,
) -> Tuple[np.ndarray]:

    initial_mae = np.mean(np.abs(y))
    print("actual mae in data", initial_mae)
    convergence_checker = get_convergence_checker(
        convergence_method=convergence_method,
        optimization_iterations=optimization_iterations,
    )
    bw = get_bw_factor(data_length=len(y), bw_multiplier=bw_multiplier)
    mae_iterations = []
    nonvariance_iterations = []
    kl_iterations = []
    
    print("prior kde")
    prior_mu, m_mesh = y_kde(y, bw, integration_mesh_size)
    prior_v, v_mesh = s2_kde(s2, bw, integration_mesh_size)
    print("int prior mu")
    int_m = integrate.simps(
        y=prior_mu,
        x=m_mesh
    )
    print(int_m)
    print("int prior v")
    int_v = integrate.simps(
        y=prior_v,
        x=v_mesh
    )
    print(int_v)

    print("denoms")
    denoms = calc_denominator(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=ensemble_size,
    )
    print("mae")
    mae = expected_mae(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=ensemble_size,
        denominators=denoms,
        ensemble_size=ensemble_size,
    )
    mae_iterations.append(mae)
    print("nonvar")
    nonvariance = expected_nonvariance(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=ensemble_size,
        denominators=denoms,
    )
    nonvariance_iterations.append(nonvariance)

    iteration = 0
    previous_prior_mu = None
    kl_iterations.append(None)
    print("iteration", iteration)
    print(f"expected mae {mae}")
    print(f"expected nonvariance {nonvariance}")
    save_iteration_stats(
        iteration=iteration,
        save_dir=save_dir,
        mae_iterations=mae_iterations,
        nonvariance_iterations=nonvariance_iterations,
        kl_iterations=kl_iterations,
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
    )

    while convergence_checker.is_not_converged(
        iteration=iteration,
        current_distribution=prior_mu,
        previous_distribution=previous_prior_mu,
    ):
        iteration += 1
        print("prior update")
        previous_prior_mu = prior_mu
        prior_mu, prior_v = update_separate_priors(
            prior_mu=prior_mu,
            prior_v=prior_v,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            denominators=denoms,
        )
        print("int prior mu")
        int_m = integrate.simps(
            y=prior_mu,
            x=m_mesh
        )
        print(int_m)
        print("int prior v")
        int_v = integrate.simps(
            y=prior_v,
            x=v_mesh
        )
        print(int_v)

        print("denoms")
        denoms = calc_denominator(
            prior_mu=prior_mu,
            prior_v=prior_v,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
        )
        print("mae")
        mae = expected_mae(
            prior_mu=prior_mu,
            prior_v=prior_v,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            denominators=denoms,
            ensemble_size=ensemble_size,
        )
        mae_iterations.append(mae)
        print("nonvariance")
        nonvariance = expected_nonvariance(
            prior_mu=prior_mu,
            prior_v=prior_v,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            denominators=denoms,
        )
        nonvariance_iterations.append(nonvariance)
        print("kl")
        kl = kl_divergence(
            prior_mu=prior_mu,
            previous_prior_mu=previous_prior_mu,
            m_mesh=m_mesh,
        )
        kl_iterations.append(kl)

        print("iteration", iteration)
        print(f"expected mae {mae}")
        print(f"expected nonvariance {nonvariance}")
        print("kl divergence from previous", kl)
        save_iteration_stats(
            iteration=iteration,
            save_dir=save_dir,
            mae_iterations=mae_iterations,
            nonvariance_iterations=nonvariance_iterations,
            kl_iterations=kl_iterations,
            prior_mu=prior_mu,
            prior_v=prior_v,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
        )
    
    projection_sizes = [1, 2, 3]+list(range(5, 30, 5)) + list(range(30, 101, 10))
    if ensemble_size not in projection_sizes:
        projection_sizes.append(ensemble_size)
        projection_sizes.sort()
    projected_mae = []
    marginal_mae = []
    for size in projection_sizes:
        print("size", size)
        print("project mae")
        p_mae = expected_mae(
            prior_mu=prior_mu,
            prior_v=prior_v,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=size,
            denominators=denoms,
            ensemble_size=ensemble_size,
        )
        projected_mae.append(p_mae)
        if size < ensemble_size:
            marginal_mae.append(None)
        elif size == ensemble_size:
            marginal_mae.append(initial_mae)
        else:
            print("marginal mae")
            m_mae = expected_marginal_mae(
                prior_mu=prior_mu,
                prior_v=prior_v,
                m_mesh=m_mesh,
                v_mesh=v_mesh,
                y=y,
                s2=s2,
                n=size - ensemble_size,
                ensemble_size=ensemble_size,
                denominators=denoms
            )
            marginal_mae.append(m_mae)
        save_projection(
            save_dir=save_dir,
            projection_sizes=projection_sizes,
            projected_mae=projected_mae,
            marginal_mae=marginal_mae,
            nonvariance=nonvariance
        )


