import numpy as np
from scipy.stats import norm

##### BASE CLASS TO CALCULATE EXPANSION POLYNOMIALS #####


PHI = lambda x: norm.cdf(x)


def calculate_sigma_coefficients(T, sigma_0, weights, ivs):
    sqT = np.sqrt(T)
    S0 = 0.5 * sigma_0 * sqT

    Hi = 0.5 * ivs * sqT
    Ei = np.exp(0.5 * (S0**2 - Hi**2))
    sigma_0 = 2.0 / sqT * norm.ppf(np.dot(weights, PHI(Hi)))
    sigma_2 = 1.0 / 2 / sqT * (-1.0 / S0 + np.dot(weights, Ei / Hi))
    S2 = sigma_0 * sigma_2 * T
    sigma_4 = (
        1.0
        / 8
        / sqT
        * (
            1.0 / S0**3 * (1 + 6 * S2 + S0**2 * (-7 - 6 * S2 + 3 * S2**2))
            + np.dot(weights, Ei / Hi**3 * (-1 + 7 * Hi**2))
        )
    )
    S4 = sigma_0 * sigma_4 * T
    sigma_6 = (
        1.0
        / 32
        / sqT
        * (
            1.0
            / S0**5
            * (
                -3
                - 45 * S2
                + S0**2 * (90 * S2 + 60 * S4)
                + S0**4 * S2 * (45 * S2 + 60 * S4 - 15 * S2**2)
                + 16 * S0**2
                - 90 * S2**2
                - 31 * S0**4
                - 45 * S0**2 * S2**2
                - S0**4 * (15 * S2 + 60 * S4)
                + 15 * S2**3 * S0**2
            )
            + np.dot(weights, Ei / Hi**5 * (3 - 16 * Hi**2 + 31 * Hi**4))
        )
    )
    return sigma_2, sigma_4, sigma_6


def get_sigma_0(T, weights, ivs):
    Hi = 0.5 * ivs * np.sqrt(T)
    return 2.0 / np.sqrt(T) * norm.ppf(np.dot(weights, PHI(Hi)))


def get_sigma_approx(m, T, sigma_0, weights, ivs, order=6):
    sigma_2, sigma_4, sigma_6 = calculate_sigma_coefficients(T, sigma_0, weights, ivs)
    if order == 2:
        return sigma_0 + sigma_2 / 2.0 * m**2
    elif order == 4:
        return sigma_0 + sigma_2 / 2.0 * m**2 + sigma_4 / 24.0 * m**4
    elif order == 6:
        return sigma_0 + sigma_2 / 2.0 * m**2 + sigma_4 / 24.0 * m**4 + sigma_6 / 720.0 * m**6
