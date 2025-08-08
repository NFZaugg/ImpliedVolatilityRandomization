from math import factorial

import numpy as np

##### BASE CLASS TO CALCULATE COLLOCATION POINTS #####


def get_gram_matrix(n, eta):
    m = np.zeros((n + 1, n + 1))

    for idx, _ in np.ndenumerate(m):
        m[idx] = eta ** sum(idx) * get_moment(sum(idx))

    return m


def get_gram_matrix_log_normal(n, params):
    mu, eta = params
    m = np.zeros((n + 1, n + 1))

    for idx, _ in np.ndenumerate(m):
        ind = sum(idx)
        m[idx] = np.exp(ind * mu + 0.5 * ind**2 * eta)

    return m


def get_gram_matrix_gamma(n, params):
    k, theta = params
    m = np.zeros((n + 1, n + 1))

    for idx, _ in np.ndenumerate(m):
        ind = sum(idx)
        m[idx] = theta**ind * np.prod([k + i for i in range(ind)])

    return m


def get_gram_matrix_beta(n, params):
    alpha, beta = params
    m = np.zeros((n + 1, n + 1))

    for idx, _ in np.ndenumerate(m):
        ind = sum(idx)
        m[idx] = np.prod([(alpha + i) / (alpha + beta + i) for i in range(ind)])

    return m


def get_moment(n):
    if n % 2 == 0:
        return double_factorial(n - 1)
    return 0


def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)


def get_alphas(r):
    return [r[j, j + 1] / r[j, j] - r[j - 1, j] / r[j - 1, j - 1] for j in range(len(r) - 1)]


def get_betas(r):
    return [(r[j + 1, j + 1] / r[j, j]) ** 2 for j in range(0, len(r) - 2)]


def get_gram_matrix_uniform(n, eta):
    u, d = eta
    return 1 / (np.linspace(np.ones(n + 1), np.ones(n + 1) * (n + 1), n + 1) + np.arange(n + 1))


def get_col_points(n, parameters, type="n"):
    if type == "n":
        m = get_gram_matrix(n, parameters[1])
    elif type == "ln":
        m = get_gram_matrix_log_normal(n, parameters)
    elif type == "u":
        m = get_gram_matrix_uniform(n, parameters)
    elif type == "gamma":
        m = get_gram_matrix_gamma(n, parameters)
    elif type == "beta":
        m = get_gram_matrix_beta(n, parameters)
    r = np.linalg.cholesky(m).T
    a = get_alphas(r)
    b = get_betas(r)
    j = np.diag(a) + np.diag(np.sqrt(b), 1) + np.diag(np.sqrt(b), -1)
    x, W = np.linalg.eig(j)
    w = W[0, :] ** 2
    idx = np.argsort(x)
    w = w[idx]
    x = x[idx]
    return w, x


if __name__ == "__main__":
    a = get_gram_matrix_beta(2, [0.5, 0.5])
    b = 1
