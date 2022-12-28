import matplotlib.pyplot as plt
import numpy as np


def bspline_basis(x, p, k, compute_derivatives=False, **kwargs):
    """Computes Cox-de Boor recursive formula for B-spline basis functions.

    Parameters
    ----------
    x : float
        The value at which to compute the basis function.
    p : int
        The order of the basis function.
    k : array-like
        The knot vector.
    compute_derivatives : bool, optional
        Whether to compute the derivative of the basis function.

    Returns
    -------
    float
        The value of the basis function.
    """

    if not p:
        return np.where(np.all([k[:-1] <= x, x < k[1:]], axis=0), 1.0, 0.0)

    b_p_minus_1 = bspline_basis(x, p - 1, k, _recursing=True)

    numer_1 = x - k[:-p]
    denom_1 = k[p:] - k[:-p]

    numer_2 = k[(p + 1) :] - x
    denom_2 = k[(p + 1) :] - k[1:-p]

    # Adjust numerators if computing derivatives
    if compute_derivatives and not kwargs.get("_recursing"):
        numer_1 = p
        numer_2 = -p

    with np.errstate(divide="ignore", invalid="ignore"):
        term_1 = np.where(denom_1 != 0, (numer_1 / denom_1), 0.0)
        term_2 = np.where(denom_2 != 0, (numer_2 / denom_2), 0.0)

    return term_1[:-1] * b_p_minus_1[:-1] + term_2 * b_p_minus_1[1:]


def plot(p, k, num=1000):
    x_min = np.min(k)
    x_max = np.max(k)

    x = np.linspace(x_min, x_max, num=num)
    N = np.array([bspline_basis(i, p, k) for i in x]).T

    for n in N:
        plt.plot(x, n)

    return plt.show()
