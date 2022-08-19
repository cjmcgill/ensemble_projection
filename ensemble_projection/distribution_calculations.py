import os
from typing import Tuple

import numpy as np
import scipy
from scipy import integrate
from scipy.special import erf

from ensemble_projection.utils import get_stats, get_bw_factor
from ensemble_projection.convergence import get_convergence_checker
from ensemble_projection.kde import y_kde, s2_kde
from ensemble_projection.likelihood import get_likelihood

def calc_denominator(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, likelihood, integration_func):
    """2d integrate Im * Iv * L, for a given i"""
    denominators = []
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        v_slices = []
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_v[i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(y=likelihood_v * p_v * prior_mu, dx=dm)
            v_slices.append(slice)
        denom = integration_func(y=v_slices, x=v_mesh)
        denominators.append(denom)
    return denominators


def calc_denominator_2d(prior_2d, m_mesh, v_mesh, y, s2, n, likelihood, integration_func):
    """2d integrate I * L, for a given i"""
    denominators = []
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        v_slices = []
        for i in range(len(v_mesh)):
            p_v = prior_2d[:, i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(y=likelihood_v * p_v, dx=dm)
            v_slices.append(slice)
        denom = integration_func(y=v_slices, x=v_mesh)
        denominators.append(denom)
    return denominators


def expected_nonvariance(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, likelihood, integration_func):
    """2d integrate abs(m) * Im * Iv * L, for a given i"""
    expected_errors = []
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(i, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        denom = denominators[j]
        v_slices = []
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_v[i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=np.abs(m_mesh) * likelihood_v * p_v * prior_mu,
                dx=dm,
            )
            v_slices.append(slice)
        numer = integration_func(y=v_slices, x=v_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def expected_nonvariance_2d(prior_2d, m_mesh, v_mesh, y, s2, n, denominators, likelihood, integration_func):
    """2d integrate abs(m) * Im * Iv * L, for a given i"""
    expected_errors = []
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        denom = denominators[j]
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        v_slices = []
        for i in range(len(v_mesh)):
            p_v = prior_2d[:, i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=np.abs(m_mesh) * likelihood_v * p_v,
                dx=dm,
            )
            v_slices.append(slice)
        numer = integration_func(y=v_slices, x=v_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def beta_error_2d(v_mesh, m_mesh, n):
    """Returns the value across m_mesh for a given v"""
    # beta=np.square(sample_m)*n/v_mesh/2
    # f=np.exp(-beta) / np.sqrt(np.pi * beta)+scipy.special.erf(np.sqrt(beta))
    v = np.reshape(v_mesh, [1, -1])
    m = np.reshape(m_mesh, [-1, 1])
    f = np.sqrt(2 * v / (n * np.pi)) \
        * np.exp(-n * m**2 / (2 * v)) \
        + m * erf(m * np.sqrt(n / (2 * v)))
    return f


def expected_mae(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, ensemble_size, likelihood, integration_func):
    """2d integrate abs(m) * beta * Im * Iv * L, for a given i"""
    expected_errors = []
    beta = beta_error_2d(v_mesh=v_mesh, m_mesh=m_mesh, n=n)
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        denom = denominators[j]
        v_slices = []
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_v[i]
            beta_v = beta[:, i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=beta_v * likelihood_v * p_v * prior_mu,
                dx=dm,
            )
            v_slices.append(slice)
        numer = integration_func(y=v_slices, x=v_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def expected_mae_2d(prior_2d, m_mesh, v_mesh, y, s2, n, denominators, ensemble_size, likelihood, integration_func):
    """2d integrate abs(m) * beta * Im * Iv * L, for a given i"""
    expected_errors = []
    beta = beta_error_2d(v_mesh=v_mesh, m_mesh=m_mesh, n=n)
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        denom = denominators[j]
        v_slices = []
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_2d[:, i]
            beta_v = beta[:, i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=beta_v * likelihood_v * p_v,
                dx=dm,
            )
            v_slices.append(slice)
        numer = integration_func(y=v_slices, x=v_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def beta_marginal_error_2d(v_mesh, m_mesh, n, y, ensemble_size):
    """Returns the value across m_mesh for a given v"""
    v = np.reshape(v_mesh, [1, -1])
    m = np.reshape(m_mesh, [-1, 1])
    sub_v = v * (n / (n + ensemble_size))**2
    sub_m = (y * ensemble_size + m * n) / (ensemble_size + n)
    f = np.sqrt(2 * sub_v / (n * np.pi)) \
        * np.exp(-n * sub_m**2 / (2 * sub_v)) \
        + sub_m * erf(sub_m * np.sqrt(n / (2 * sub_v)))
    return f


def expected_marginal_mae(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, ensemble_size, likelihood, integration_func):
    """2d integrate abs(m) * beta * Im * Iv * L, for a given i"""
    expected_errors = []
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, ensemble_size, likelihood)
        beta = beta_marginal_error_2d(v_mesh, m_mesh, n, sample_y, ensemble_size)
        denom = denominators[j]
        v_slices = []
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_v[i]
            beta_v = beta[:, i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=beta_v * likelihood_v * p_v * prior_mu,
                dx=dm,
            )
            v_slices.append(slice)
        numer = integration_func(y=v_slices, x=v_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def expected_marginal_mae_2d(prior_2d, m_mesh, v_mesh, y, s2, n, denominators, ensemble_size, likelihood, integration_func):
    """2d integrate abs(m) * beta * Im * Iv * L, for a given i"""
    expected_errors = []
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, ensemble_size, likelihood)
        beta = beta_marginal_error_2d(v_mesh, m_mesh, n, sample_y, ensemble_size)
        denom = denominators[j]
        v_slices = []
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_2d[:, i]
            beta_v = beta[:, i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=beta_v * likelihood_v * p_v,
                dx=dm,
            )
            v_slices.append(slice)
        numer = integration_func(y=v_slices, x=v_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)



def update_separate_priors(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, likelihood, learning_rate, integration_func):
    new_prior_mu = update_prior_mu(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=n,
        denominators=denominators,
        likelihood=likelihood,
        learning_rate=learning_rate,
        integration_func=integration_func,
    )
    new_prior_v = update_prior_v(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=n,
        denominators=denominators,
        likelihood=likelihood,
        learning_rate=learning_rate,
        integration_func=integration_func,
    )
    return new_prior_mu, new_prior_v


def update_prior_mu(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, likelihood, learning_rate, integration_func):
    """1d integrate Im * Iv * L over v, for a given i"""
    new_prior_mu = np.zeros_like(prior_mu)
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        denom = denominators[j]
        m_slices = np.zeros_like(prior_mu)
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            p_m = prior_mu[i]
            likelihood_m = sample_l[i]
            slice = integration_func(
                y=likelihood_m * prior_v * p_m,
                x=v_mesh,
            )
            m_slices[i] = slice / denom / len(y)
        new_prior_mu = new_prior_mu + m_slices
    if learning_rate != 1.0:
        new_prior_mu = (new_prior_mu / prior_mu) ** learning_rate * prior_mu

    return new_prior_mu


def update_prior_v(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, likelihood, learning_rate, integration_func):
    """1d integrate Im * Iv * L over mu, for a given i"""
    new_prior_v = np.zeros_like(prior_v)
    dm = m_mesh[1] - m_mesh[0]
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        denom = denominators[j]
        v_slices = np.zeros_like(prior_v)
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_v[i]
            likelihood_v = sample_l[:, i]
            slice = integration_func(
                y=likelihood_v * prior_mu * p_v,
                dx=dm,
            )
            v_slices[i] = slice / denom / len(y)
        new_prior_v = new_prior_v + v_slices
    if learning_rate != 1.0:
        new_prior_v = (new_prior_v / prior_v) ** learning_rate * prior_v

    return new_prior_v


def update_prior_2d(prior_2d, m_mesh, v_mesh, y, s2, n, denominators, likelihood, learning_rate, integration_func):
    new_prior_2d = np.zeros_like(prior_2d)
    for j in range(len(y)):
        denom = denominators[j]
        sample_y = y[j]
        sample_s2 = s2[j]
        sample_l = get_likelihood(j, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        new_prior_2d = new_prior_2d + sample_l * prior_2d / denom / len(y)
    if learning_rate != 1.0:
        new_prior_2d = (new_prior_2d / prior_2d) ** learning_rate * prior_2d
    return new_prior_2d


def integrate_2d(prior_2d, m_mesh, v_mesh, integration_func):
    v_slices = np.zeros_like(v_mesh)
    dm = m_mesh[1] - m_mesh[0]
    for i in range(len(v_mesh)):
        v = v_mesh[i]
        p_v = prior_2d[:, i]
        slice = integration_func(
            y=p_v,
            dx = dm,
        )
        v_slices[i] = slice
    integration = integration_func(
        y=v_slices,
        x=v_mesh
    )
    return integration


def kl_divergence(prior_mu, previous_prior_mu, m_mesh, integration_func, threshold=1e-20) -> float:
    """Check the divergence between the prior distributions between iterations"""
    prior_mu[prior_mu<threshold] = threshold
    previous_prior_mu[previous_prior_mu<threshold] = threshold
    kl = integration_func(
        y=prior_mu * np.log(prior_mu / previous_prior_mu),
        x=m_mesh,
    )
    return kl


def kl_divergence_2d(prior_2d, previous_prior_2d, m_mesh, v_mesh, integration_func, threshold=1e-20) -> float:
    """Check the divergence between the prior distributions between iterations"""
    prior_2d[prior_2d<threshold] = threshold
    previous_prior_2d[previous_prior_2d<threshold] = threshold
    dm = m_mesh[1] - m_mesh[0]
    v_slices = []
    for i in range(len(v_mesh)):
        v = v_mesh[i]
        prior_v = prior_2d[:, i]
        previous_v = previous_prior_2d[:, i]
        slice = integration_func(
            y=prior_v * np.log(prior_v / previous_v),
            dx=dm,
        )
        v_slices.append(slice)
    kl = integration_func(y=v_slices, x=v_mesh)
    return kl

