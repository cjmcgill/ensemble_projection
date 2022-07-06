import numpy as np
import scipy


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


def kde_2d(y, s2, bw, mu_mesh_size, v_mesh_size):
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
