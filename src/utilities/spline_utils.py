# import SEAL as SE
import numpy as np
# np.set_printoptions(precision=4)
import torch


def find_index(x, t):  # TODO: Provide initial guess for speed-up?
    """
    Given a knot vector t, find the index mu such that t[mu] <= x < t[mu+1]
    :param x: Parameter value
    :param t: knot vector
    :return:
    """
    if abs(x - t[-1]) <= 1.0e-14:
        # at endpoint, return last non trivial index
        for i in range(len(t) - 1, 0, -1):
            if t[i] < x:
                return i
    for i in range(len(t) - 1):
        if t[i] <= x < t[i + 1]:
            return i


def create_knots(a, b, p, n):
    """
    Returns a p+1 regular knot vector starting at a and ending at b with total length of n + p + 1.
    """
    t = np.lib.pad(np.linspace(a, b, num=n - p + 1), (p, p), mode='edge')
    return t


def evaluate_non_zero_basis_splines(x, mu, t, p):
    b = 1
    for k in range(1, p + 1):
        # extract relevant knots
        t1 = t[mu - k + 1: mu + 1]
        t2 = t[mu + 1: mu + k + 1]
        # append 0 to end of first term, and insert 0 to start of second term
        # noinspection PyArgumentList
        omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1), where=((t2 - t1) != 0))
        b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)

    return b


def bspline_collocation_matrix(d=1, n_in=128, n_out=512):
    """
    Generate the n_out x n_in collocation matrix of n_in uniform B-spline basis functions of degree d evaluated
    at n_out evenly spaced parameter values.

    :param int d    : Degree of the splines
    :param int n_in : Number of uniform B-spline basis functions
    :param int n_out: Number of evenly spaced parameter values for evaluation
    :return: Univariate collocation matrix as a 2-dimensional array
    """
    # Parameter values
    x = np.linspace(0, 1, n_out)

    # Knots
    t = create_knots(0, 1, d, n_in)

    # Spline space
    # knots_x = SE.create_knots(0, 1, d, n_in)
    # T = SE.SplineSpace(d, knots_x)
    # fs = T.basis

    col_mat = [np.concatenate([np.array((find_index(x0, t)-d)*[0]),
                               evaluate_non_zero_basis_splines(x0, find_index(x0, t), t, d),
                               np.array((n_in - 1 - find_index(x0, t))*[0])]) for x0 in x]
    col_mat = np.array(col_mat)

    # Create a collocation matrix
    # col_mat = [[fs[i0](x0)[0] for i0 in range(n_in)] for x0 in x]
    # col_mat = np.array(col_mat)
    col_mat = torch.tensor(col_mat, dtype=torch.float).cuda()

    return col_mat


def evaluate_from_col_mat(coefficients, col_mat):
    """
    Given a univariate collocation matrix 'col_mat' and bivariate coefficient array 'coefficients' (with additional
    batch size dimension). Let f be the function given as the linear combination of the coefficients with the bivariate
    tensor-product basis of the univariate basis underlying the collocation matrix. Evaluate f at the pairs of parameter
    values underlying the collocation matrix using a two-stage tensor contraction, which is memory and computationally
    efficient.

    :param array coefficients: Trivariate array of size |batch| x |basis| x |basis|
    :param array col_mat     : Univariate collocation matrix of size |parameters| x |basis|
    :return: Evaluated functions as an array of size |batch| x |parameters| x |parameters|
    """
    z = torch.einsum('ik,mkl->mil', col_mat, coefficients)
    z = torch.einsum('jl,mil->mij', col_mat, z)

    return z


def get_level_set_from_coefficients(coefficients, col_mat):
    """
    Given a univariate collocation matrix 'col_mat' and bivariate coefficient array 'coefficients' (with additional
    batch size dimension). Let f be the function given as the linear combination of the coefficients with the bivariate
    tensor-product basis of the univariate basis underlying the collocation matrix. Compute:
     * |batch| x |parameters| x |parameters| array z of values of f at the pairs of parameter values underlying the
       collocation matrix,
     * |batch| x |parameters| x |parameters| binary array z_treshold representing the sign of z, i.e., with 1 at
       nonnegative values of z and 0 otherwise.

    :param array coefficients: Trivariate array of size |batch| x |basis| x |basis|
    :param array col_mat     : Univariate collocation matrix of size |parameters| x |basis|
    :return: |batch| x |parameters| x |parameters|
    """
    z = evaluate_from_col_mat(coefficients, col_mat)

    # For some reason this threshold is the inverse of the original done in the generation of the data. Not sure why.
    z_treshold = z.cpu().detach().numpy()
    z_treshold[z_treshold > 0] = 1
    z_treshold[z_treshold < 0] = 0

    return z, z_treshold
