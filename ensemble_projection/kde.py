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
