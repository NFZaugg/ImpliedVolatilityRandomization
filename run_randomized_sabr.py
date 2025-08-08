from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from general.hagan import hagan_implied_volatility
from general.util import imply_volatility
from randomizations.rand_sabr import RandSABR

"""
Runs the SABR randomization on the SPX option quotes in the Data folder. 
We plot both the exact randomization values, as well as the 6th order expansion. 
We see that the 6th order is not sufficient for this case, as the approximation diverges from the exact solution.

However, the fit of the randomized SABR is significantly better due to the extra flexiblity.

"""


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True
if __name__ == "__main__":
    r = 0.05
    spot = 5535
    today = date(2024, 7, 31)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))

    expiryDates = [date(2024, 8, 16), date(2024, 9, 20), date(2024, 10, 18), date(2024, 11, 15)]
    months = ["August", "September", "October", "November"]
    rand_sabr_params = np.array(
        [
            [0.9, 0.33414235, -0.6907052, 1.70409171, 1.4284071],
            [0.9, 0.31866989, -0.6858314, 3.63637799, 0.48022903],
            [0.9, 0.31660493, -0.6753167, 2.89472607, 0.46295823],
            [0.9, 0.33841544, -0.68947279, 4.72176393, 0.28050375],
        ]
    )
    sabr_params = np.array(
        [
            [0.9, 0.33795777, -0.54409678, 3.94972169],
            [0.9, 0.32457636, -0.59188021, 2.16977728],
            [0.9, 0.3280161, -0.47201946, 1.65175483],
            [0.9, 0.34742342, -0.56569068, 1.5388523],
        ]
    )
    for i, ax, eDate, month, param, sabr_params in zip(
        range(4), axs.flatten(), expiryDates, months, rand_sabr_params, sabr_params
    ):
        data = pd.read_csv(f"Data/{month}.csv", index_col=0).squeeze()

        t = (eDate - today).days / 365
        k, iv = np.array(data.index), data.values
        randomized_sabr = RandSABR(params_rand=param, n_col_points=2)

        # For the plot we prefer a uniform grid
        k_uni = np.linspace(k[0] / spot, k[-1] / spot, 100) * spot

        # Get the randomized prices and expansion IVS
        randomized_prices = randomized_sabr.prices(spot, k_uni, t, r)
        # 6th order ivs
        randomized_ivs = randomized_sabr.ivs(spot, k_uni, t, r)
        # Plotting
        ax.plot(k / spot, 100 * iv, "x", label="Market Quote", markersize=7, color="sandybrown")
        ax.plot(
            k_uni / spot,
            100 * imply_volatility(randomized_prices, spot, k_uni, t, r, 0.4),
            label="Randomized SABR",
            marker="^",
            linewidth=1.5,
            color="#00539C",
            markevery=5,
        )
        ax.plot(
            k_uni / spot,
            100 * np.array(randomized_ivs),
            label="Randomized SABR (6th-order approximation)",
            marker="*",
            linewidth=1.5,
            color="purple",
            markevery=5,
        )

        ax.plot(
            k_uni / spot,
            100 * hagan_implied_volatility(k_uni, t, spot * np.exp(r * t), *sabr_params),
            label="Regular SABR",
            marker="o",
            linestyle="dashdot",
            linewidth=1.5,
            color=plt.cm.viridis(0.5),
            markevery=5,
        )
        # Adjust Plot
        ax.grid()
        ax.set_xlabel("Strike (relative to ATM) ", size=20)
        ax.set_ylabel("Implied Volatility [\%]", size=20)
        ax.set_title(f"{month} (TTM {np.round(t,2)}y)", size=20)
        ax.set_xlim([k[0] / spot, k[-1] / spot])
        l = ax.legend(prop={"size": 10}, fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")
    plt.suptitle("SPX Option Chain 2024-07-31", fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Plots/randomized_sabr_with_data.png", dpi=300)
    plt.show()
