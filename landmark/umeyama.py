"""
    Fork from https://github.com/clementinboittiaux/umeyama-python
    Usage:
        from umeyama import umeyama
        c, R, t = umeyama(X, Y)
"""

import numpy as np


def umeyama(X, Y, est_c=True):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)

    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1

    if est_c:
        var_x = np.square(X - mu_x).sum(axis=0).mean()
        c = np.trace(np.diag(D) @ S) / var_x
    else:
        c = 1
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t