import numpy as np
import scipy
from scipy import integrate

from ensemble_projection.likelihood import get_likelihood


def get_initial_prior_func(initial_prior):
    if initial_prior == "kde":
        return prior_kde
    elif initial_prior == "gaussian":
        return gaussian_1d
    elif initial_prior == "likelihood":
        return normalized_likelihood_1d
    elif initial_prior == "uniform":
        return uniform_1d


def get_initial_prior_func_2d(initial_prior):
    if initial_prior == "kde":
        return kde_2d
    elif initial_prior == "likelihood":
        return normalized_likelihood_2d
    elif initial_prior == "gaussian":
        return gaussian_2d
    elif initial_prior == "uniform":
        return uniform_2d


def y_kde(y, bw, mesh_size):
    m_mesh = get_mesh_values(y, bw, mesh_size)
    m_dist = np.zeros(mesh_size)
    for i in range(mesh_size):
        m_dist[i] = manual_kde(
            sample_x=m_mesh[i],
            x_mean=y,
            x_std=np.std(y),
            bw=bw
        )
    return m_dist, m_mesh


def s2_kde(s2, bw, mesh_size):
    ls2 = np.log(s2)
    lv_mesh = get_mesh_values(ls2, bw, mesh_size)
    lv_dist = np.zeros(mesh_size)
    for i in range(mesh_size):
        lv_dist[i] = manual_kde(
            sample_x=lv_mesh[i],
            x_mean=ls2,
            x_std=np.std(ls2),
            bw=bw,
        )
    v_mesh = np.exp(lv_mesh)
    v_dist = lv_dist / v_mesh
    return v_dist, v_mesh


def get_meshes(y, s2, bw, mu_mesh_size, v_mesh_size):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    lv_mesh = get_mesh_values(np.log(s2), bw, v_mesh_size)
    v_mesh = np.exp(lv_mesh)
    return m_mesh, v_mesh


def get_mesh_values(x, bw, mesh_size):
    bw_x = np.std(x) * bw
    max_val = max(x) + 2 * bw_x
    min_val = min(x) - 2 * bw_x
    mesh = np.linspace(min_val, max_val, mesh_size)
    return mesh


def manual_kde(sample_x,x_mean,x_std,bw): #for a loop
    p=1/(np.sqrt(2*np.pi)*x_std*bw) * \
        np.exp(-1/2*np.square((sample_x-x_mean)/x_std/bw))
    return np.mean(p)


def cum_kde(sample_mu,mu,mu_std,bw):
    p=1/2*(1+scipy.special.erf((sample_mu-mu)/np.sqrt(2)/mu_std/bw))
    return np.mean(p)


def kde_2d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    m_expanded = np.reshape(m_mesh, [-1, 1])
    y_std = np.std(y)
    ls2 = np.log(s2)
    ls2_std = np.std(ls2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    lv_expanded = np.reshape(lv_mesh, [1, -1])
    lv_prior = np.zeros([mu_mesh_size, v_mesh_size])
    data_size = len(y)
    for i in range(data_size):
        p = 1 / (2 * np.pi * bw**2 * ls2_std * y_std) \
            * np.exp(-1 / 2 * np.square((m_expanded - y[i]) / y_std / bw)) \
            * np.exp(-1 / 2 * np.square((lv_expanded - ls2[i]) / ls2_std / bw))
        lv_prior = lv_prior + p / data_size
    v_mesh = np.exp(lv_mesh)
    prior = lv_prior / np.reshape(v_mesh, [1, -1])
    return m_mesh, v_mesh, prior


def normalized_likelihood_2d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    dm = m_mesh[1] - m_mesh[0]
    ls2 = np.log(s2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    v_mesh = np.exp(lv_mesh)
    prior = np.zeros([mu_mesh_size, v_mesh_size])
    data_size = len(y)

    for i in range(data_size):
        sample_y = y[i]
        sample_s2 = s2[i]
        sample_l = get_likelihood(i, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        v_slices = []
        for j in range(v_mesh_size):
            slice = integration_func(
                y=sample_l[:, j],
                dx=dm,
            )
            v_slices.append(slice)
        denom = integration_func(
            y=v_slices,
            x=v_mesh,
        )
        prior = prior + sample_l / data_size / denom

    return m_mesh, v_mesh, prior


def gaussian_2d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    m_expanded = np.reshape(m_mesh, [-1, 1])
    y_std = np.std(y)
    ls2 = np.log(s2)
    ls2_std = np.std(ls2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    lv_expanded = np.reshape(lv_mesh, [1, -1])
    lv_prior = 1 / (2 * np.pi * bw_multiplier**2 * ls2_std * y_std) \
        * np.exp(-1 / 2 * np.square((m_expanded - 0) / y_std / bw_multiplier)) \
        * np.exp(-1 / 2 * np.square((lv_expanded - np.mean(ls2)) / ls2_std / bw_multiplier))
    v_mesh = np.exp(lv_mesh)
    prior = lv_prior / np.reshape(v_mesh, [1, -1])
    return m_mesh, v_mesh, prior


def prior_kde(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    prior_mu, m_mesh = y_kde(y, bw, mu_mesh_size)
    prior_v, v_mesh = s2_kde(s2, bw, v_mesh_size)
    return prior_mu, prior_v, m_mesh, v_mesh


def gaussian_1d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    y_std = np.std(y)
    ls2 = np.log(s2)
    ls2_std = np.std(ls2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    lv_prior = 1/ np.sqrt(2 * np.pi) / (bw_multiplier * ls2_std) \
        * np.exp(-1 / 2 * np.square((lv_mesh - np.mean(ls2)) / ls2_std / bw_multiplier))
    prior_mu = 1 / np.sqrt(2 * np.pi) / ( bw_multiplier * y_std) \
        * np.exp(-1 / 2 * np.square((m_mesh - 0) / y_std / bw_multiplier))

    v_mesh = np.exp(lv_mesh)
    prior_v = lv_prior / v_mesh
    
    return prior_mu, prior_v, m_mesh, v_mesh


def normalized_likelihood_1d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    dm = m_mesh[1] - m_mesh[0]
    y_std = np.std(y)
    ls2 = np.log(s2)
    ls2_std = np.std(ls2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    data_size = len(y)

    for i in range(data_size):
        sample_y = y[i]
        sample_s2 = s2[i]
        sample_l = get_likelihood(i, m_mesh, v_mesh, sample_y, sample_s2, n, likelihood)
        v_slices = []
        for j in range(v_mesh_size):
            slice = integration_func(
                y=sample_l[:, j],
                dx=dm,
            )
            v_slices.append(slice)
        denom = integration_func(
            y=v_slices,
            x=lv_mesh,
        )
        prior_lv = np.zeros(v_mesh_size)
        v_slices = np.zeros(v_mesh_size)
        for j in range(v_mesh_size):
            slice = integration_func(
                y=sample_l[:, j],
                dx=dm
            )
            v_slices[j] = slice
        prior_lv = prior_lv + v_slices / data_size / denom

        prior_mu = np.zeros(mu_mesh_size)
        m_slices = np.zeros(mu_mesh_size)
        for j in range(mu_mesh_size):
            slice = integration_func(
                y=sample_l[j],
                x=lv_mesh
            )
            m_slices[j] = slice
        prior_mu = prior_mu + m_slices / data_size / denom

    v_mesh = np.exp(lv_mesh)
    prior_v = prior_lv / v_mesh
    
    return prior_mu, prior_v, m_mesh, v_mesh


def uniform_2d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    m_expanded = np.reshape(m_mesh, [-1, 1])
    y_std = np.std(y)
    ls2 = np.log(s2)
    ls2_std = np.std(ls2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    lv_expanded = np.reshape(lv_mesh, [1, -1])
    lv_prior = np.full([mu_mesh_size, v_mesh_size], 1 / (mu_mesh_size * v_mesh_size))
    v_mesh = np.exp(lv_mesh)
    prior = lv_prior / np.reshape(v_mesh, [1, -1])
    return m_mesh, v_mesh, prior


def uniform_1d(y, s2, bw, mu_mesh_size, v_mesh_size, n, likelihood, bw_multiplier, integration_func):
    m_mesh = get_mesh_values(y, bw, mu_mesh_size)
    y_std = np.std(y)
    ls2 = np.log(s2)
    ls2_std = np.std(ls2)
    lv_mesh = get_mesh_values(ls2, bw, v_mesh_size)
    lv_prior = np.full([v_mesh_size], 1 / (lv_mesh[-1] - lv_mesh[0]))
    prior_mu = np.full([mu_mesh_size], 1 / (m_mesh[-1] - m_mesh[0]))
    v_mesh = np.exp(lv_mesh)
    prior_v = lv_prior / v_mesh
    
    return prior_mu, prior_v, m_mesh, v_mesh