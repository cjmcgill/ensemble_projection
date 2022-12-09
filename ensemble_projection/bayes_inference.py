from distutils.log import error
import os
from typing import Callable, Tuple

import numpy as np
from scipy import integrate

"""new change"""

from ensemble_projection.utils import get_stats, get_bw_factor, \
    save_iteration_stats, save_iteration_stats_2d, save_projection
from ensemble_projection.convergence import get_convergence_checker
from ensemble_projection.kde import get_initial_prior_func, get_initial_prior_func_2d, get_meshes, kde_2d, y_kde, s2_kde
from ensemble_projection.distribution_calculations import calc_denominator, calc_denominator_2d, \
    expected_mae_2d, expected_marginal_mae, expected_marginal_mae_2d, expected_nonvariance, \
    expected_mae, expected_nonvariance_2d, integrate_2d, update_prior_2d, update_separate_priors, \
    kl_divergence, kl_divergence_2d
from ensemble_projection.likelihood import persistent_likelihood, remove_likelihood


def bayes_infer(
    target_path: str,
    preds_path: str,
    ensemble_size: int,
    save_dir: str,
    convergence_method: str = "iteration_count",
    optimization_iterations: int = 10,
    no_bessel_correction_needed: bool = False,
    truncate_data_length: int = None,
    bw_multiplier: float = 1.,
    mu_mesh_size: int = 1025,
    v_mesh_size: int = 129,
    prior_method: str = 'combined',
    max_projection_size: int = 100,
    save_iteration_steps: bool = False,
    initial_prior: str = "kde",
    learning_rate: float = 1.0,
    error_basis: bool = False,
    integration_method: str = "trapz",
    kl_threshold: float = 1e-5,
    fraction_change_threshold: float = 1e-4,
    scratch_dir: str = None,
    likelihood_calculation: str = "persistent"
):
    """
    Perform a bayesian inference calculations for the error of a NN
    prediction without variance error included. Use this information
    to quantify the variance and nonvariance error and project the
    total error with different numbers of ensemble models used.
    """

    ids, ensemble_vars, errors = get_stats(
        target_path=target_path,
        preds_path=preds_path,
        ensemble_size=ensemble_size,
        truncate_data_length=truncate_data_length,
        no_bessel_correction=no_bessel_correction_needed,
        error_basis=error_basis,
    )
    if scratch_dir is None:
        scratch_dir = os.path.join(save_dir, "scratch")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    integration_func = {
        "simps": integrate.simps,
        "trapz": integrate.trapz,
    }[integration_method]
    inference_func = {
        "combined": combined_prior,
        "separate": separate_priors,
    }[prior_method]
    inference_func(
        y=errors,
        s2=ensemble_vars,
        ensemble_size=ensemble_size,
        convergence_method=convergence_method,
        optimization_iterations=optimization_iterations,
        save_dir=save_dir,
        bw_multiplier=bw_multiplier,
        mu_mesh_size=mu_mesh_size,
        v_mesh_size=v_mesh_size,
        max_projection_size=max_projection_size,
        save_iteration_steps=save_iteration_steps,
        initial_prior=initial_prior,
        learning_rate=learning_rate,
        error_basis=error_basis,
        integration_func=integration_func,
        kl_threshold=kl_threshold,
        fraction_change_threshold=fraction_change_threshold,
        scratch_dir=scratch_dir,
        likelihood_calculation=likelihood_calculation,
    )
    if likelihood_calculation == "persistent":
        remove_likelihood(scratch_dir=scratch_dir)


def separate_priors(
    y: np.ndarray,
    s2: np.ndarray,
    ensemble_size: int,
    convergence_method: str,
    optimization_iterations: int,
    save_dir: str,
    integration_func: Callable,
    scratch_dir: str,
    bw_multiplier: float = 1.,
    mu_mesh_size: int = 1025,
    v_mesh_size: int = 129,
    max_projection_size: int = 100,
    save_iteration_steps: bool = False,
    initial_prior: str = "kde",
    learning_rate: float = 1.0,
    error_basis: bool = False,
    kl_threshold: float = 1e-5,
    fraction_change_threshold: float = 1e-4,
    likelihood_calculation: str = "persistent",
) -> Tuple[np.ndarray]:

    initial_mae = np.mean(np.abs(y))
    print("actual mae in data", initial_mae)
    convergence_checker = get_convergence_checker(
        convergence_method=convergence_method,
        optimization_iterations=optimization_iterations,
        kl_threshold=kl_threshold,
        fraction_change_threshold=fraction_change_threshold,
    )
    initial_prior_func = get_initial_prior_func(initial_prior=initial_prior)
    bw = get_bw_factor(data_length=len(y), bw_multiplier=bw_multiplier, dimension=1)

    m_mesh, v_mesh = get_meshes(y, s2, bw, mu_mesh_size, v_mesh_size)

    if likelihood_calculation == "persistent":
        print("likelihood")
        likelihood = persistent_likelihood(
            scratch_dir=scratch_dir,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
        )
    else:
        likelihood = None

    print("prior kde")
    prior_mu, prior_v, m_mesh, v_mesh =initial_prior_func(y, s2, bw, mu_mesh_size, v_mesh_size, ensemble_size, likelihood, bw_multiplier, integration_func)

    if save_iteration_steps:
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
        likelihood=likelihood,
        integration_func=integration_func
    )

    mae_iterations = []
    nonvariance_iterations = []
    if save_iteration_steps:
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
            likelihood=likelihood,
            integration_func=integration_func
        )
    else:
        mae = None
    mae_iterations.append(mae)
    
    if save_iteration_steps or convergence_method == "fraction_change_threshold":
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
            likelihood=likelihood,
            integration_func=integration_func
        )
    else:
        nonvariance = None
    nonvariance_iterations.append(nonvariance)
    previous_nonvariance = None
    previous_prior_mu = None

    iteration = 0

    kl_iterations = []
    kl_iterations.append(None)
    print("iteration", iteration)

    print(f"expected mae {mae}")
    print(f"expected nonvariance {nonvariance}")
    save_iteration_stats(
        iteration=iteration,
        scratch_dir=scratch_dir,
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
        current_nonvariance=nonvariance,
        previous_nonvariance=previous_nonvariance,
        kl=kl
    ):
        iteration += 1
        print("prior update")
        previous_nonvariance = nonvariance
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
            likelihood=likelihood,
            learning_rate=learning_rate,
            integration_func=integration_func,
        )

        if save_iteration_steps:
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
            likelihood=likelihood,
            integration_func=integration_func,
        )

        if save_iteration_steps:
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
                likelihood=likelihood,
                integration_func=integration_func,
            )
        else:
            mae = None
        mae_iterations.append(mae)
        
        if save_iteration_steps or convergence_method == "fraction_change_threshold":
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
                likelihood=likelihood,
                integration_func=integration_func,
            )
        else:
            nonvariance = None
        nonvariance_iterations.append(nonvariance)

        if save_iteration_steps or convergence_method == "kl_threshold":
            print("kl")
            kl = kl_divergence(
                prior_mu=prior_mu,
                previous_prior_mu=previous_prior_mu,
                m_mesh=m_mesh,
                integration_func=integration_func,
            )

        else:
            kl = None
        kl_iterations.append(kl)

        print("iteration", iteration)
        print("kl divergence from previous", kl)
        print(f"expected mae {mae}")
        print(f"expected nonvariance {nonvariance}")
        save_iteration_stats(
            iteration=iteration,
            scratch_dir=scratch_dir,
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
    projection_sizes = [i for i in projection_sizes if i <= max_projection_size]
    if ensemble_size not in projection_sizes:
        projection_sizes.append(ensemble_size)
        projection_sizes.sort()
    projected_mae = []
    marginal_mae = []
    nonvariance = expected_nonvariance(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=ensemble_size,
        denominators=denoms,
        likelihood=likelihood,
        integration_func=integration_func,
    )
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
            likelihood=likelihood,
            integration_func=integration_func,
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
                denominators=denoms,
                likelihood=likelihood,
                integration_func=integration_func,
            )
            marginal_mae.append(m_mae)
        save_projection(
            save_dir=save_dir,
            projection_sizes=projection_sizes,
            projected_mae=projected_mae,
            marginal_mae=marginal_mae,
            nonvariance=nonvariance
        )


def combined_prior(
    y: np.ndarray,
    s2: np.ndarray,
    ensemble_size: int,
    convergence_method: str,
    optimization_iterations: int,
    save_dir: str,
    integration_func: Callable,
    scratch_dir: str,
    bw_multiplier: float = 1.,
    mu_mesh_size: int = 1025,
    v_mesh_size: int = 129,
    max_projection_size: int = 100,
    save_iteration_steps: bool = False,
    initial_prior: str = "kde",
    learning_rate: float = 1.0,
    error_basis: bool = False,
    kl_threshold: float = 1e-5,
    fraction_change_threshold: float = 1e-4,
    likelihood_calculation: str = "persistent",
) -> Tuple[np.ndarray]:

    initial_mae = np.mean(np.abs(y))
    print("actual mae in data", initial_mae)
    convergence_checker = get_convergence_checker(
        convergence_method=convergence_method,
        optimization_iterations=optimization_iterations,
        kl_threshold=kl_threshold,
        fraction_change_threshold=fraction_change_threshold,
    )
    bw = get_bw_factor(data_length=len(y), bw_multiplier=bw_multiplier, dimension=2)
    initial_prior_func_2d = get_initial_prior_func_2d(initial_prior=initial_prior)
    kl_iterations = []
    
    m_mesh, v_mesh = get_meshes(y, s2, bw, mu_mesh_size, v_mesh_size)

    if likelihood_calculation == "persistent":
        print("likelihood")
        likelihood = persistent_likelihood(
            scratch_dir=scratch_dir,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
        )
    else:
        likelihood = None

    print("initial_prior")
    m_mesh, v_mesh, prior_2d = initial_prior_func_2d(
        y=y,
        s2=s2,
        bw=bw,
        mu_mesh_size=mu_mesh_size,
        v_mesh_size=v_mesh_size,
        n=ensemble_size,
        likelihood=likelihood,
        bw_multiplier=bw_multiplier,
        integration_func=integration_func,
    )
    if save_iteration_steps:
        print("int prior")
        int_prior = integrate_2d(
            prior_2d=prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            integration_func=integration_func,
        )
        print(int_prior)

    print("denoms")
    denoms = calc_denominator_2d(
        prior_2d=prior_2d,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=ensemble_size,
        likelihood=likelihood,
        integration_func=integration_func,
    )

    mae_iterations = []
    nonvariance_iterations = []
    if save_iteration_steps:
        print("mae")
        mae = expected_mae_2d(
            prior_2d = prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            denominators=denoms,
            ensemble_size=ensemble_size,
            likelihood=likelihood,
            integration_func=integration_func,
        )
    else:
        mae = None
    mae_iterations.append(mae)

    if save_iteration_steps or convergence_method == "fraction_change_threshold":
        print("nonvar")
        nonvariance = expected_nonvariance_2d(
            prior_2d=prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            denominators=denoms,
            likelihood=likelihood,
            integration_func=integration_func,
        )
    else:
        nonvariance = None
    nonvariance_iterations.append(nonvariance)

    iteration = 0
    previous_prior_2d = None
    previous_nonvariance = None
    kl = None
    kl_iterations.append(kl)
    print("iteration", iteration)

    print(f"expected mae {mae}")
    print(f"expected nonvariance {nonvariance}")
    save_iteration_stats_2d(
        iteration=iteration,
        scratch_dir=scratch_dir,
        save_dir=save_dir,
        mae_iterations=mae_iterations,
        nonvariance_iterations=nonvariance_iterations,
        kl_iterations=kl_iterations,
        prior_2d=prior_2d,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
    )

    while convergence_checker.is_not_converged(
        iteration=iteration,
        current_nonvariance=nonvariance,
        previous_nonvariance=previous_nonvariance,
        kl=kl
    ):
        iteration += 1
        print("prior update")
        previous_nonvariance = nonvariance
        previous_prior_2d = prior_2d
        prior_2d = update_prior_2d(
            prior_2d=prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            denominators=denoms,
            likelihood=likelihood,
            learning_rate=learning_rate,
            integration_func=integration_func,
        )
        if save_iteration_steps:
            print("int prior")
            int_prior = integrate_2d(
                prior_2d=prior_2d,
                m_mesh=m_mesh,
                v_mesh=v_mesh,
                integration_func=integration_func,
            )
            print(int_prior)

        print("denoms")
        denoms = calc_denominator_2d(
            prior_2d=prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=ensemble_size,
            likelihood=likelihood,
            integration_func=integration_func,
        )

        if save_iteration_steps:
            print("mae")
            mae = expected_mae_2d(
                prior_2d=prior_2d,
                m_mesh=m_mesh,
                v_mesh=v_mesh,
                y=y,
                s2=s2,
                n=ensemble_size,
                denominators=denoms,
                ensemble_size=ensemble_size,
                likelihood=likelihood,
                integration_func=integration_func,
            )
        else:
            mae = None
        mae_iterations.append(mae)
        
        if save_iteration_steps or convergence_method == "fraction_change_threshold":
            print("nonvariance")
            nonvariance = expected_nonvariance_2d(
                prior_2d=prior_2d,
                m_mesh=m_mesh,
                v_mesh=v_mesh,
                y=y,
                s2=s2,
                n=ensemble_size,
                denominators=denoms,
                likelihood=likelihood,
                integration_func=integration_func,
            )
        else:
            nonvariance = None
        nonvariance_iterations.append(nonvariance)

        if save_iteration_steps or convergence_method == "kl_threshold":
            print("kl")
            kl = kl_divergence_2d(
                prior_2d=prior_2d,
                previous_prior_2d=previous_prior_2d,
                m_mesh=m_mesh,
                v_mesh=v_mesh,
                integration_func=integration_func,
            )
        else:
            kl = None
        kl_iterations.append(kl)

        print("iteration", iteration)
        print("kl divergence from previous", kl)
        print(f"expected mae {mae}")
        print(f"expected nonvariance {nonvariance}")
        save_iteration_stats_2d(
            iteration=iteration,
            scratch_dir=scratch_dir,
            save_dir=save_dir,
            mae_iterations=mae_iterations,
            nonvariance_iterations=nonvariance_iterations,
            kl_iterations=kl_iterations,
            prior_2d=prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
        )
    
    projection_sizes = [1, 2, 3]+list(range(5, 30, 5)) + list(range(30, 101, 10))
    projection_sizes = [i for i in projection_sizes if i <= max_projection_size]
    if ensemble_size not in projection_sizes:
        projection_sizes.append(ensemble_size)
        projection_sizes.sort()
    projected_mae = []
    marginal_mae = []
    nonvariance = expected_nonvariance_2d(
        prior_2d=prior_2d,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=ensemble_size,
        denominators=denoms,
        likelihood=likelihood,
        integration_func=integration_func,
    )
    for size in projection_sizes:
        print("size", size)
        print("project mae")
        p_mae = expected_mae_2d(
            prior_2d=prior_2d,
            m_mesh=m_mesh,
            v_mesh=v_mesh,
            y=y,
            s2=s2,
            n=size,
            denominators=denoms,
            ensemble_size=ensemble_size,
            likelihood=likelihood,
            integration_func=integration_func,
        )
        projected_mae.append(p_mae)
        if size < ensemble_size:
            marginal_mae.append(None)
        elif size == ensemble_size:
            marginal_mae.append(initial_mae)
        else:
            print("marginal mae")
            m_mae = expected_marginal_mae_2d(
                prior_2d=prior_2d,
                m_mesh=m_mesh,
                v_mesh=v_mesh,
                y=y,
                s2=s2,
                n=size - ensemble_size,
                ensemble_size=ensemble_size,
                denominators=denoms,
                likelihood=likelihood,
                integration_func=integration_func,
            )
            marginal_mae.append(m_mae)
        save_projection(
            save_dir=save_dir,
            projection_sizes=projection_sizes,
            projected_mae=projected_mae,
            marginal_mae=marginal_mae,
            nonvariance=nonvariance
        )