from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from general.util import imply_volatility
from randomizations.rand_flat import RandomizedFlat

"""
This simple example runs the flat randomization to produce a volatility smile with two parameters. 

"""

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True

if __name__ == "__main__":
    # Parameters
    r = 0.02
    spot = 100
    t = 2
    k = np.linspace(0.4, 2.5, 100) * spot
    m = np.log(k / spot) + r * t

    # Randomization parameters (mu,nu)
    rand_params = [[-0.9, 0.1], [-1.5, 0.06]]
    # Plotting tools
    fig, axs = plt.subplots(ncols=2, figsize=(14, 6))
    linestyle = ["solid", "dotted", "dashed", "dashdot", "dotted"]
    markers = ["none", "o", "none", "s", "^"]

    # Run two randomization with parameters as
    for i, params in enumerate(rand_params):
        randomization = RandomizedFlat(params_rand=params, n_col_points=4)
        prices = randomization.prices(spot, k, t, r)

        # Reference implied volatilites using Brent
        ivs_p = imply_volatility(prices, spot, k, t, r, 0.3)

        # Expansion IVs for different orders
        ivs_approx2 = np.array(randomization.ivs(spot, k, t, r, 2))
        ivs_approx4 = np.array(randomization.ivs(spot, k, t, r, 4))
        ivs_approx6 = np.array(randomization.ivs(spot, k, t, r, 6))

        # Plot all the results
        axs[i].set_title(r"Randomized Flat Volatility: $(\mu,\nu)=$" + f"{params[0],params[1]}", size=20)
        axs[i].set_xlabel(r"$m = \log (S_0/K) + rT $", size=20)
        axs[i].set_ylabel("Implied Volatility [\%]", size=20)
        axs[i].plot(
            m,
            ivs_p * 100,
            marker=markers[3],
            linestyle=linestyle[0],
            color="#00539C",
            markevery=5,
            label="IV (Brent Method)",
        )
        axs[i].plot(
            m,
            ivs_approx2 * 100,
            color=plt.cm.viridis(1),
            linestyle=linestyle[2],
            marker=markers[0],
            markevery=5,
            label="2nd-Order Expansion",
        )
        axs[i].plot(
            m,
            ivs_approx4 * 100,
            color=plt.cm.viridis(0.8),
            linestyle=linestyle[3],
            marker=markers[0],
            markevery=5,
            label="4th-Order Expansion",
        )
        axs[i].plot(
            m,
            ivs_approx6 * 100,
            color=plt.cm.viridis(0.6),
            linestyle=linestyle[1],
            marker=markers[0],
            markevery=5,
            label="6th-Order Expansion",
        )
        axs[i].grid()
        l = axs[i].legend(prop={"size": 15}, fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")
    plt.tight_layout()
    plt.savefig("Plots/randomized.flat.png", dpi=300)
    plt.show()
