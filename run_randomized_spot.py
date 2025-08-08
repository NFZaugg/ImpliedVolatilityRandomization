import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from general.util import imply_volatility
from randomizations.rand_spot import RandomizedSpot

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Liberation Serif"]

plt.rcParams["text.usetex"] = True

"""
This example runs the spot randomization to produce a volatility smile from a bimodal distributiopn. 

"""

if __name__ == "__main__":
    # Parameters
    r = 0.02
    spot = 3
    t = 1
    k = np.linspace(0.8, 1.4, 100) * spot
    m = np.log(k / spot) + r * t
    sigma = 0.12
    etas = np.linspace(0.1, 0.2, 3)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))
    axs = axs.T
    linestyle = ["dotted", "dotted", "dotted", "dashdot", "dotted"]
    markers = ["o", "o", "o", "s", "^"]
    for i, z in enumerate(zip(etas, linestyle, markers)):
        eta, ls, ma = z
        randomization = RandomizedSpot(params_rand=[eta], n_col_points=2)
        prices = randomization.prices(spot, k, t, r, sigma)

        # Reference implied volatilites using Brent
        ivs_p = imply_volatility(prices, spot, k, t, r, 0.3)

        # Expansion terms
        vol2, vol3, vol4 = randomization.ivs(spot, k, t, r, sigma)

        #RNPD
        dk = k[1] - k[0]
        axs[i, 0].plot(
            k[2:] / spot,
            np.diff((np.diff(prices)) / dk) / dk,
            label=f"$\\nu$ = {np.round(eta,2)}",
            marker=ma,
            linewidth=1.5,
            linestyle=ls,
            color="#00539C",
            markevery=5,
        )

        axs[i, 0].grid()
        axs[i, 0].set_xlabel("Strike (relative to ATM) ", size=20)
        axs[i, 0].set_ylabel(r"Probability Density $P(S_T = x)$ ", size=20)
        l = axs[i, 0].legend(prop={"size": 20}, fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")

        axs[i, 1].plot(
            m,
            ivs_p * 100,
            label=f"IV (Brent Method)",
            marker="o",
            linewidth=1.5,
            linestyle=ls,
            color=plt.cm.viridis(1),
            markevery=5,
        )
        axs[i, 1].plot(
            m,
            vol2 * 100,
            label=f"2nd-Order Expansion",
            marker="none",
            linewidth=1.5,
            color=plt.cm.viridis(0.11),
            linestyle=ls,
            markevery=5,
        )
        axs[i, 1].plot(
            m,
            vol3 * 100,
            label=f"3rd-Order Expansion",
            marker="^",
            linewidth=1.5,
            color=plt.cm.viridis(0.4),
            linestyle=ls,
            markevery=5,
        )
        axs[i, 1].plot(
            m,
            vol4 * 100,
            label=f"4th-Order Expansion",
            marker="s",
            linewidth=1.5,
            color=plt.cm.viridis(0.7),
            linestyle=ls,
            markevery=5,
        )
        axs[i, 1].grid()
        axs[i, 1].set_xlabel(r"m = $\log(S/K) +rT$ ", size=20)
        axs[i, 1].set_ylabel("Implied Volatility [\%]", size=20)
        l = axs[i, 1].legend(prop={"size": 15}, loc="lower center", fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")

    plt.tight_layout()
    plt.savefig("Plots/RandomizedSpot_flat", dpi=300)
    plt.show()
