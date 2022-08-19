import os

import numpy as np

def calc_likelihood(mu, v, sample_y, sample_s2, n):
    """Returns the likelihood for a particular datapoint to be returned by a given mu or v."""
    expanded_mu = np.reshape(mu, [-1, 1])
    expanded_v = np.reshape(v, [1, -1])
    p= np.power( expanded_v * 2 * np.pi, -n / 2) * np.exp(-1 / 2 / expanded_v * (n * np.square(sample_y - expanded_mu) + sample_s2 * (n-1)))
    return p


def persistent_likelihood(m_mesh, v_mesh, y, s2, n, scratch_dir):
    """Returns a memmap array for the likelihood function of shape(data, mu, v)."""
    likelihood_path = os.path.join(scratch_dir, "likelihood.dat")
    likelihood_map = np.memmap(
        filename=likelihood_path,
        dtype=float,
        mode="w+",
        shape=(len(y), len(m_mesh), len(v_mesh))
    )
    for j in range(len(y)):
        sample_y = y[j]
        sample_s2 = s2[j]
        likelihood_map[j] = calc_likelihood(m_mesh, v_mesh, sample_y, sample_s2, n)
    del likelihood_map
    likelihood_map = np.memmap(
        filename=likelihood_path,
        dtype=float,
        mode="r",
        shape=(len(y), len(m_mesh), len(v_mesh))
    )
    return likelihood_map


def get_likelihood(idx, m_mesh, v_mesh, y, s2, n, likelihood):
    if isinstance(likelihood, np.memmap):
        return likelihood[idx]
    else:
        return calc_likelihood(m_mesh, v_mesh, y, s2, n)


def remove_likelihood(scratch_dir):
    likelihood_path = os.path.join(scratch_dir, "likelihood.dat")
    os.remove(likelihood_path)
