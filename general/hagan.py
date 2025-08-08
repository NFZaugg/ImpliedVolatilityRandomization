import numpy as np
from scipy.optimize import minimize, newton


def calibration_hagan(K, t, iv, f, beta, iv_atm):
    K = np.array(K)
    iv = np.array(iv)
    # x = [rho, gamma]
    f_obj = lambda x: target_val(beta, x[0], x[1], iv, K, f, t, iv_atm, f)[0]

    # Random initial guess
    initial = np.array([-0.5, 0.6])
    pars = minimize(
        f_obj, initial, bounds=[(-1, 1), (0, 100)], method="nelder-mead", options={"disp": False, "xatol": 1e-9}
    )
    rho_est = pars.x[0]
    gamma_est = pars.x[1]
    alpha_est = target_val(beta, rho_est, gamma_est, iv, K, f, t, iv_atm, f)[1]
    return np.array([beta, alpha_est, rho_est, gamma_est])


def determine_optimal_alpha(iv_atm, k_atm, t, f, beta, rho, gamma):
    target = lambda alpha: hagan_implied_volatility([k_atm], t, f, beta, alpha, rho, gamma).squeeze() - iv_atm
    # Initial guess does not really matter here
    alpha_est = newton(target, 1.05, tol=1e-7)
    return alpha_est


def target_val(beta, rho, gamma, iv, K, f, t, iv_atm, k_atm):
    # Error is defined as a difference between the market and the model
    alpha_est = determine_optimal_alpha(iv_atm, k_atm, t, f, beta, rho, gamma)
    error_vector = hagan_implied_volatility(K, t, f, beta, alpha_est, rho, gamma) - iv

    # Target value is a norm of the error_vector
    value = np.linalg.norm(error_vector)
    return value, alpha_est


def hagan_implied_volatility(K, T, f, beta, alpha, rho, gamma):
    # We make sure that the input is of array type
    if type(K) == float:
        K = np.array([K])
    if type(K) is not np.ndarray:
        K = np.array(K).reshape([len(K), 1])
    z = gamma / alpha * np.power(f * K, (1.0 - beta) / 2.0) * np.log(f / K)
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    A = alpha / (
        np.power(f * K, ((1.0 - beta) / 2.0))
        * (
            1.0
            + np.power(1.0 - beta, 2.0) / 24.0 * np.power(np.log(f / K), 2.0)
            + np.power((1.0 - beta), 4.0) / 1920.0 * np.power(np.log(f / K), 4.0)
        )
    )
    B1 = (
        1.0
        + (
            np.power((1.0 - beta), 2.0) / 24.0 * alpha * alpha / (np.power((f * K), 1 - beta))
            + 1 / 4 * (rho * beta * gamma * alpha) / (np.power((f * K), ((1.0 - beta) / 2.0)))
            + (2.0 - 3.0 * rho * rho) / 24.0 * gamma * gamma
        )
        * T
    )
    imp_vol = A * (z / x_z) * B1
    B2 = (
        1.0
        + (
            np.power(1.0 - beta, 2.0) / 24.0 * alpha * alpha / (np.power(f, 2.0 - 2.0 * beta))
            + 1.0 / 4.0 * (rho * beta * gamma * alpha) / np.power(f, (1.0 - beta))
            + (2.0 - 3.0 * rho * rho) / 24.0 * gamma * gamma
        )
        * T
    )

    # Special treatment for ATM strike price
    if len(np.where(K == f)[0]) > 0:
        imp_vol[np.where(K == f)] = alpha / np.power(f, (1 - beta)) * B2
    return imp_vol.squeeze()
