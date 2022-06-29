import os
from typing import Tuple

import numpy as np
import scipy
from scipy import integrate
from scipy.special import erf

from ensemble_projection.utils import get_stats, get_bw_factor
from ensemble_projection.convergence import get_convergence_checker
from ensemble_projection.kde import y_kde, s2_kde

def likelihood(mu, v, sample_y, sample_s2, n):
    """Returns the likelihood for a particular datapoint to be returned by a given mu or v."""
    p= np.power( v * 2 * np.pi, -n / 2) * np.exp(-1 / 2 / v * (n * np.square(sample_y - mu) + sample_s2 * (n-1)))
    return p


def persistent_likelihood(save_dir, m_mesh, v_mesh, y, s2, n):
    """Returns a memmap array for the likelihood function of shape(data, mu, v)."""
    likelihood_path = os.path.join(save_dir, "scratch", "likelihood.dat")
    likelihood_map = np.memmap(
        filename=likelihood_path,
        dtype=float,
        mode="w+",
        shape=(len(y), len(m_mesh), len(v_mesh))
    )
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            likelihood_m = likelihood(m, v_mesh, sample_y, sample_s2, n)
            likelihood_map[j, i] = likelihood_m
    del likelihood_map
    likelihood_map = np.memmap(
        filename=likelihood_path,
        dtype=float,
        mode="r",
        shape=(len(y), len(m_mesh), len(v_mesh))
    )
    return likelihood_map

def calc_denominator(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n):
    """2d integrate Im * Iv * L, for a given i"""
    denominators = []
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        m_slices = []
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            p_m = prior_mu[i]
            likelihood_m = likelihood(m, v_mesh, sample_y, sample_s2, n)
            slice = integrate.trapz(y=likelihood_m * prior_v * p_m, x=v_mesh)
            m_slices.append(slice)
        denom = integrate.trapz(y=m_slices, x=m_mesh)
        denominators.append(denom)
    return denominators


def expected_nonvariance(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators):
    """2d integrate abs(m) * Im * Iv * L, for a given i"""
    expected_errors = []
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        denom = denominators[j]
        m_slices = []
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            p_m = prior_mu[i]
            likelihood_m = likelihood(m, v_mesh, sample_y, sample_s2, n)
            slice = integrate.trapz(
                y=np.abs(m) * likelihood_m * prior_v * p_m,
                x=v_mesh,
            )
            m_slices.append(slice)
        numer = integrate.trapz(y=m_slices, x=m_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def beta_error(v_mesh, sample_m, n):
    """Returns the value across m_mesh for a given v"""
    # beta=np.square(sample_m)*n/v_mesh/2
    # f=np.exp(-beta) / np.sqrt(np.pi * beta)+scipy.special.erf(np.sqrt(beta))
    f = np.sqrt(2 * v_mesh / (n * np.pi)) \
        * np.exp(-n * sample_m**2 / (2 * v_mesh)) \
        + sample_m * erf(sample_m * np.sqrt(n / (2 * v_mesh)))
    return f


def expected_mae(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, ensemble_size):
    """2d integrate abs(m) * beta * Im * Iv * L, for a given i"""
    expected_errors = []
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        denom = denominators[j]
        m_slices = []
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            p_m = prior_mu[i]
            beta = beta_error(v_mesh, m, n)
            likelihood_m = likelihood(m, v_mesh, sample_y, sample_s2, ensemble_size)
            slice = integrate.trapz(
                y=beta * likelihood_m * prior_v * p_m,
                x=v_mesh,
            )
            m_slices.append(slice)
        numer = integrate.trapz(y=m_slices, x=m_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def beta_marginal_error(v_mesh, sample_m, n, y, ensemble_size):
    """Returns the value across m_mesh for a given v"""
    # m = (y * ensemble_size + x * n)/(ensemble_size + n)
    # v = (n/(ensemble_size + n))**2
    # beta=np.square((sample_m * n + y * ensemble_size) / (n + ensemble_size)) * n / (v_mesh * (n / (n + ensemble_size)) ** 2) / 2
    # f=np.exp(-beta) / np.sqrt(np.pi * beta)+scipy.special.erf(np.sqrt(beta))
    sub_v = v_mesh * (n / (n + ensemble_size))**2
    sub_m = (y * ensemble_size + sample_m * n) / (ensemble_size + n)
    f = np.sqrt(2 * sub_v / (n * np.pi)) \
        * np.exp(-n * sub_m**2 / (2 * sub_v)) \
        + sub_m * erf(sub_m * np.sqrt(n / (2 * sub_v)))
    return f


def expected_marginal_mae(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators, ensemble_size):
    """2d integrate abs(m) * beta * Im * Iv * L, for a given i"""
    expected_errors = []
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        denom = denominators[j]
        m_slices = []
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            p_m = prior_mu[i]
            beta = beta_marginal_error(v_mesh, m, n, sample_y, ensemble_size)
            likelihood_m = likelihood(m, v_mesh, sample_y, sample_s2, ensemble_size)
            slice = integrate.trapz(
                y=beta * likelihood_m * prior_v * p_m,
                x=v_mesh,
            )
            m_slices.append(slice)
        numer = integrate.trapz(y=m_slices, x=m_mesh)
        expected_errors.append(numer / denom)
    return np.mean(expected_errors)


def update_separate_priors(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators):
    new_prior_mu = update_prior_mu(
        prior_mu=prior_mu,
        prior_v=prior_v,
        m_mesh=m_mesh,
        v_mesh=v_mesh,
        y=y,
        s2=s2,
        n=n,
        denominators=denominators,
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
    )
    return new_prior_mu, new_prior_v


def update_prior_mu(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators):
    """1d integrate Im * Iv * L over v, for a given i"""
    new_prior_mu = np.zeros_like(prior_mu)
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        denom = denominators[j]
        m_slices = np.zeros_like(prior_mu)
        for i in range(len(m_mesh)):
            m = m_mesh[i]
            p_m = prior_mu[i]
            likelihood_m = likelihood(m, v_mesh, sample_y, sample_s2, n)
            slice = integrate.trapz(
                y=likelihood_m * prior_v * p_m,
                x=v_mesh,
            )
            m_slices[i] = slice / denom / len(y)
        new_prior_mu = new_prior_mu + m_slices

    return new_prior_mu


def update_prior_v(prior_mu, prior_v, m_mesh, v_mesh, y, s2, n, denominators):
    """1d integrate Im * Iv * L over mu, for a given i"""
    new_prior_v = np.zeros_like(prior_v)
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        denom = denominators[j]
        v_slices = np.zeros_like(prior_v)
        for i in range(len(v_mesh)):
            v = v_mesh[i]
            p_v = prior_v[i]
            likelihood_v = likelihood(m_mesh, v, sample_y, sample_s2, n)
            slice = integrate.trapz(
                y=likelihood_v * prior_mu * p_v,
                x=m_mesh,
            )
            v_slices[i] = slice / denom / len(y)
        new_prior_v = new_prior_v + v_slices

    return new_prior_v