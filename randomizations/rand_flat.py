import numpy as np
from pydantic import BaseModel

from general.collocation import get_col_points
from general.expansions import get_sigma_0, get_sigma_approx
from general.util import black_scholes


class RandomizedFlat(BaseModel):
    # Randomization class for a lognormal randomization of a flat black scholes surface
    params_rand: tuple
    n_col_points: int

    def prices(self, spot, k, t, r):
        return self._mixed_bs_prices(spot, k, t, r)

    def ivs(self, spot, k, t, r, order):
        return self._mixed_bs_ivs(spot, k, t, r, order)

    def _mixed_bs_ivs(self, spot, k, t, r, order):
        weights, colP = get_col_points(self.n_col_points, self.params_rand, "ln")
        m = np.log(spot / k) + r * t
        sigma0 = get_sigma_0(t, weights, colP)
        ivsMixed = get_sigma_approx(m, t, sigma0, weights, colP, order)
        return ivsMixed

    def _mixed_bs_prices(self, spot, k, t, r):
        weights, colP = get_col_points(self.n_col_points, self.params_rand, "ln")
        pricesForParams = [black_scholes(spot, k, t, r, sig) for sig in colP]
        return (weights[:, np.newaxis] * pricesForParams).sum(axis=0)
