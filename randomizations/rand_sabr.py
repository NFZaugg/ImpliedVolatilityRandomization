import numpy as np
from pydantic import BaseModel

from general.collocation import get_col_points
from general.expansions import get_sigma_0, get_sigma_approx
from general.hagan import hagan_implied_volatility
from general.util import black76

nCol = 2


class RandSABR(BaseModel):
    # Randomizes the gamma parameter (vol of vol) with a Gamma random variable of two parameters
    params_rand: list
    n_col_points: int

    def prices(self, spot, k, t, r):
        return self._mixed_hagan_prices(spot, k, t, r)

    def ivs(self, spot, k, t, r):
        return self._mixed_hagan_ivs(spot, k, t, r)

    def _mixed_hagan_ivs(self, spot, k, t, r):
        non_randomized_params, gamma_params = self.params_rand[:3], self.params_rand[-2:]
        weights, colP = get_col_points(nCol, gamma_params, "gamma")

        params = np.array([non_randomized_params + [c] for c in colP])
        modelIVForParams = [np.atleast_1d(hagan_implied_volatility(k, t, spot * np.exp(r * t), *p)) for p in params]
        ivsMixed = []
        for strikeN, strike in enumerate(k):
            m = np.log(spot / strike) + r * t
            ivsAtStrike = np.array([ivs[strikeN] for ivs in modelIVForParams])
            sigma0 = get_sigma_0(t, weights, ivsAtStrike)
            ivsMixed.append(get_sigma_approx(m, t, sigma0, weights, ivsAtStrike))
        return ivsMixed

    def _mixed_hagan_prices(self, spot, k, t, r):
        non_randomized_params, gamma_params = self.params_rand[:3], self.params_rand[-2:]
        weights, col_points = get_col_points(nCol, gamma_params, "gamma")

        params = np.array([non_randomized_params + [c] for c in col_points])
        f = spot * np.exp(r * t)
        pricesForParams = [black76(f, k, t, r, hagan_implied_volatility(k, t, f, *p)) for p in params]
        return (weights[:, np.newaxis] * pricesForParams).sum(axis=0)
