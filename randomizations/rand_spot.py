import numpy as np
from pydantic import BaseModel
from scipy.special import erfc
from scipy.stats import norm

from general.collocation import get_col_points
from general.util import black_scholes


class RandomizedSpot(BaseModel):
    # Randomization class for a randomization of the spot
    params_rand: tuple
    n_col_points: int

    def prices(self, spot, k, t, r, sigma):
        return self._mixed_spot_prices(spot, k, t, r, sigma)

    def ivs(self, spot, k, t, r, sigma):
        m = np.log(spot / k) + r * t
        ivs2, ivs3, ivs4 = self._get_sigma_approx(m, t, spot, sigma)
        return ivs2, ivs3, ivs4

    def _mixed_spot_prices(self, spot, k, t, r, sigma):
        eta = self.params_rand[0]
        params = (np.log(spot) - eta**2 / 2, eta**2)
        weights, colP = get_col_points(self.n_col_points, params, "ln")

        pricesForParams = [black_scholes(theta, k, t, r, sigma) for theta in colP]
        return (weights[:, np.newaxis] * pricesForParams).sum(axis=0)

    def _get_sigma_approx(self, m, t, spot, sigma):
        eta = self.params_rand[0]
        params = (np.log(spot) - eta**2 / 2, eta**2)
        weights, theta = get_col_points(self.n_col_points, params, "ln")
        sT = np.sqrt(t)

        def d1(theta, sig=sigma):
            return 0.5 * sig * sT + np.log(theta / spot) / (sig * sT)

        def d2(theta, sig=sigma):
            return -0.5 * sig * sT + np.log(theta / spot) / (sig * sT)

        sigma_0 = (
            2
            / np.sqrt(t)
            * norm.ppf(
                1
                / 2
                * (
                    1
                    + sum(
                        [
                            l * (theta / spot * norm.cdf(d1(theta)) - norm.cdf(d2(theta)))
                            for theta, l in zip(theta, weights)
                        ]
                    )
                )
            )
        )
        h_dash = sum(
            [
                l * ((theta / spot * norm.pdf(d1(theta)) - norm.pdf(d2(theta))) / (sigma * sT) + norm.cdf(d2(theta)))
                for theta, l in zip(theta, weights)
            ]
        )
        h_dash_dash = sum(
            [
                l
                * (
                    -theta
                    / spot
                    * norm.pdf(d1(theta))
                    * (np.log(theta / spot) / (sigma**3 * sT**3) + 0.5 / (sigma * sT))
                    + norm.pdf(d2(theta)) * (np.log(theta / spot) / (sigma**3 * sT**3) - 0.5 / (sigma * sT))
                    + 2 * norm.pdf(d2(theta)) / (sigma * sT)
                    - norm.cdf(d2(theta))
                )
                for theta, l in zip(theta, weights)
            ]
        )
        h_dash_dash_dash = sum(
            [
                (1 / 2) * l * erfc((t * sigma**2 - 2 * np.log(theta / spot)) / (2 * np.sqrt(2) * sT * sigma))
                - (
                    np.exp(-((t * sigma**2) / 8) - (np.log(theta / spot) ** 2) / (2 * t * sigma**2))
                    * np.sqrt(theta / spot)
                    * l
                    * (3 * t * sigma**2 + 2 * np.log(theta / spot))
                )
                / (2 * np.sqrt(2 * np.pi) * t ** (3 / 2) * sigma**3)
                for theta, l in zip(theta, weights)
            ]
        )
        h_4_dash = sum(
            [
                -(1 / 2) * l * erfc((t * sigma**2 - 2 * np.log(theta / spot)) / (2 * np.sqrt(2) * sT * sigma))
                + (
                    np.exp(-((t * sigma**2) / 8) - np.log(theta / spot) ** 2 / (2 * t * sigma**2))
                    * np.sqrt(theta / spot)
                    * l
                    * (
                        t * sigma**2 * (-4 + 7 * t * sigma**2)
                        + 8 * t * sigma**2 * np.log(theta / spot)
                        + 4 * np.log(theta / spot) ** 2
                    )
                )
                / (4 * np.sqrt(2 * np.pi) * t ** (5 / 2) * sigma**5)
                for theta, l in zip(theta, weights)
            ]
        )

        fx = norm.cdf(d2(spot, sigma_0))
        fy = sT * norm.pdf(d1(spot, sigma_0))
        fxx = -norm.cdf(d2(spot, sigma_0)) + norm.pdf(d2(spot, sigma_0)) / (sigma_0 * sT)
        fxxx = -((3 * np.exp(-(sigma_0**2 * t) / 8)) / (2 * np.sqrt(2 * np.pi) * sigma_0 * sT)) + 0.5 * erfc(
            (sigma_0 * sT) / (2 * np.sqrt(2))
        )
        fxyy = (np.exp(-((sigma_0**2 * t) / 8)) * sigma_0 * t ** (3 / 2)) / (8 * np.sqrt(2 * np.pi))
        fxxy = (np.exp(-((sigma_0**2 * t) / 8)) * (-4 + sigma_0**2 * t)) / (4 * np.sqrt(2 * np.pi) * sigma_0**2 * sT)
        fyyy = (np.exp(-((sigma_0**2 * t) / 8)) * (t ** (3 / 2)) * (-4 + sigma_0**2 * t)) / (16 * np.sqrt(2 * np.pi))
        fxy = -0.5 * sT * norm.pdf(d2(spot, sigma_0))
        fyy = -norm.pdf(d1(spot, sigma_0)) / 4 * sigma_0 * t ** (3 / 2)
        fyyyy = (3 * np.exp(-((sigma_0**2 * t) / 8)) * sigma_0 * t ** (5 / 2)) / (16 * np.sqrt(2 * np.pi)) - (
            np.exp(-((sigma_0**2 * t) / 8)) * sigma_0**3 * t ** (7 / 2)
        ) / (64 * np.sqrt(2 * np.pi))
        fxxxx = (
            -(np.exp(-((sigma_0**2 * t) / 8)) / (np.sqrt(2 * np.pi) * sigma_0**3 * t ** (3 / 2)))
            + (7 * np.exp(-((sigma_0**2 * t) / 8))) / (4 * np.sqrt(2 * np.pi) * sigma_0 * sT)
            - 0.5 * erfc((sigma_0 * sT) / (2 * np.sqrt(2)))
        )
        fxyyy = (np.exp(-((sigma_0**2 * t) / 8)) * t ** (3 / 2)) / (8 * np.sqrt(2 * np.pi)) - (
            np.exp(-((sigma_0**2 * t) / 8)) * sigma_0**2 * t ** (5 / 2)
        ) / (32 * np.sqrt(2 * np.pi))
        fyxxx = (3 * np.exp(-((sigma_0**2 * t) / 8))) / (2 * np.sqrt(2 * np.pi) * sigma_0**2 * sT) - (
            np.exp(-((sigma_0**2 * t) / 8)) * sT
        ) / (8 * np.sqrt(2 * np.pi))
        fxxyy = (
            (np.exp(-((sigma_0**2 * t) / 8)) * np.sqrt(2 / np.pi)) / (sigma_0**3 * sT)
            + (np.exp(-((sigma_0**2 * t) / 8)) * sT) / (4 * np.sqrt(2 * np.pi) * sigma_0)
            - (np.exp(-((sigma_0**2 * t) / 8)) * sigma_0 * t ** (3 / 2)) / (16 * np.sqrt(2 * np.pi))
        )
        sigma_dash = (h_dash - fx) / fy
        sigma_dash_dash = (h_dash_dash - fxx - 2 * fxy * sigma_dash - fyy * sigma_dash**2) / fy
        sigma_dash_dash_dash = (
            h_dash_dash_dash
            - fxxx
            - 3 * fxxy * sigma_dash
            - 3 * fxyy * sigma_dash**2
            - 3 * fxy * sigma_dash_dash
            - 3 * fyy * sigma_dash * sigma_dash_dash
            - fyyy * sigma_dash**3
        ) / fy
        sigma_4_dash = (
            h_4_dash
            - fxxxx
            - fyyyy * sigma_dash**4
            - 4 * fxyyy * sigma_dash**3
            - 4 * fyxxx * sigma_dash
            - 6 * fxxyy * sigma_dash**2
            - 6 * fyyy * sigma_dash**2 * sigma_dash_dash
            - 12 * fxyy * sigma_dash * sigma_dash_dash
            - 6 * fxxy * sigma_dash_dash
            - 4 * fyy * sigma_dash_dash_dash * sigma_dash
            - 6 * fyy * sigma_dash_dash
            - 4 * fxy * sigma_dash_dash_dash
        ) / fy
        vol4 = (
            sigma_0
            + sigma_dash * m
            + 0.5 * sigma_dash_dash * m**2
            + 1 / 6 * sigma_dash_dash_dash * m**3
            + 1 / 24 * sigma_4_dash * m**4
        )
        vol2 = sigma_0 + sigma_dash * m + 0.5 * sigma_dash_dash * m**2
        vol3 = sigma_0 + sigma_dash * m + 0.5 * sigma_dash_dash * m**2 + 1 / 6 * sigma_dash_dash_dash * m**3

        return vol2, vol3, vol4
