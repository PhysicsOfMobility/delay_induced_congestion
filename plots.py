import os

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt


def delaycontourplot(file, tau_file, nu_file):
    """Countour plot (Fig. 2a) of in-rate vs. delay from Mathematica generated csv files. Run
    Mathematica notebook to generate csv files or load delivered csv files
    from disk.

    Parameters
    -----------
    file : str 
            path to csv file containing stability information
    tau_file : str
                path to csv file containing the delay times tau
    nu_file : str
            path to csv file containing the in-rate values
            
    Returns 
    -------
    """

    data = pd.read_csv(file, header=None)
    tau_data = pd.read_csv(tau_file, header=None)
    nu_data = pd.read_csv(nu_file, header=None)

    time_vals = data.to_numpy()
    tau_tvals = tau_data.to_numpy()
    nu_tvals = nu_data.to_numpy()

    bool_times = time_vals <= 0
    bool_times = np.ma.masked_where(~bool_times, bool_times)

    tau_vals = np.arange(0, 20.5, 0.5)
    nu_in_vals = np.arange(1, 1.31, 0.01)

    _, ax = plt.subplots()

    cmap = colors.ListedColormap(["white"])
    cmap.set_bad(color="darkgrey")

    ax.imshow(
        bool_times,
        extent=[tau_vals[0], tau_vals[-1], nu_in_vals[0], nu_in_vals[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    ax.plot(tau_tvals, nu_tvals)

    markers = [(0.0, 1.1), (4, 1.1), (8, 1.1)]
    x, y = zip(*markers)
    ax.plot(x, y, "o", c="red")

    ax.set_yticks(np.arange(1, 1.4, 0.1))
    ax.set_xticks(np.arange(0, 25, 5))
    ax.set_yticklabels(np.array([1, 1.1, 1.2, 1.3]), fontsize=14)
    ax.set_xticklabels(np.arange(0, 25, 5), fontsize=14)
    ax.set_xlabel("delay", fontsize=14)
    ax.set_ylabel("in-rate", fontsize=14)


def combinedcontourplot(file, avfile, tau_file, nu_file, av_taufile, av_nufile):
    """Plot contourplot for delay vs. in-rate for the case with time-averaged informatin (Fig. 3a).
    Use Mathematica generated csv files.

    Parameters
    -----------
        file : str
            Path to csv file containing stability information for non-averaged case
        avfile : str
                Path to csv file containing stability information for averaged case
        tau_file : str
                Path to csv file containing the delay times tau for non-averaged case
        nu_file : str
               Path to csv file containing the in-rate values for non-averaged case
        av_taufile : str 
                Path to csv file containing the delay times tau for averaged case
        av_nufile : str 
                Path to csv file containing the in-rate values for averaged case
    
    Returns 
    -------
    """
    av_tau_vals = pd.read_csv(av_taufile, header=None).to_numpy()
    av_stab_vals = pd.read_csv(av_nufile, header=None).to_numpy()
    tau_vals = pd.read_csv(tau_file, header=None).to_numpy()
    stab_vals = pd.read_csv(nu_file, header=None).to_numpy()
    av_time_vals = pd.read_csv(avfile, header=None).to_numpy()
    time_vals = pd.read_csv(file, header=None).to_numpy()

    colorarray = np.zeros((32, 41))
    for i in range(0, 32):
        for j in range(0, 41):
            av_md = av_time_vals[i, j]
            md = time_vals[i, j]

            if av_md > 0 and md > 0:
                colorarray[i, j] = 1
            if av_md > 0 and md < 0:
                colorarray[i, j] = 0.25
            if av_md < 0 and md > 0:
                colorarray[i, j] = 0.6
            if av_md < 0 and md < 0:
                colorarray[i, j] = 0

    _, ax = plt.subplots()
    cmap = colors.ListedColormap(["white", "red", "blue", "darkgrey"])
    ax.imshow(
        colorarray,
        extent=[0, 20, 1, 1.31],
        origin="lower",
        aspect="auto",
        cmap=cmap,
    )
    ax.plot(tau_vals, stab_vals, "-", color="black")
    ax.plot(av_tau_vals, av_stab_vals, "--", color="green")

    ax.set_yticks(np.arange(1, 1.4, 0.1))
    ax.set_xticks(np.arange(0, 25, 5))
    ax.set_yticklabels(np.array([1, 1.1, 1.2, 1.3]), fontsize=14)
    ax.set_xticklabels(np.arange(0, 25, 5), fontsize=14)
    ax.set_xlabel("delay", fontsize=14)
    ax.set_ylabel("in-rate", fontsize=14)


def plot_crit_inrates_time_averaging(tau1file, tau5file, tau10file):
    """Plot critical in-rates vs. averaging time window at which congestion occurs for different
    time delays. Uses Mathematica generate csv files. Run

    Parameters
    ----------
    tau1file : str
                Path to csv file containing data for time delay 1
    tau5file : str
                Path to csv file containing data for time delay 5
    tau10file : str
                Path to csv file containing data for time delay 10
    """

    time_vals = np.arange(1, 51, 1)
    nu_vals = np.arange(1, 1.301, 0.001)

    def stab_line_finder(file, time_vals, nu_vals):
        """Find the critical in-rates for each averaging time from Mathematica calculation results.
        
        Parameters 
        ----------
        file : str
            Path to csv file containing stability information
        time_vals : array-like
                    averaging time values
        nu_vals : array-like
                    in-rate values
        
        Returns 
        -------
        array-like
        """
        
        tau_vals = pd.read_csv(file, header=None)
        tau_vals = tau_vals.to_numpy()

        instab_bound = []
        for time_idx in range(0, len(time_vals)):
            nu_idx = np.where(tau_vals[:, time_idx] > 0)[0][0]
            instab_bound.append(nu_vals[nu_idx])

        return instab_bound

    instab_bound_10 = stab_line_finder(tau10file, time_vals, nu_vals)
    instab_bound_5 = stab_line_finder(tau5file, time_vals, nu_vals)
    instab_bound_1 = stab_line_finder(tau1file, time_vals, nu_vals)

    _, ax = plt.subplots()

    ax.plot(time_vals, instab_bound_10, label=r"delay $\tau=10$")
    ax.plot(time_vals, instab_bound_5, label=r"delay $\tau=5$")
    ax.plot(time_vals, instab_bound_1, label=r"delay $\tau=1$")

    ax.axhline(instab_bound_10[0], linestyle="dotted")
    ax.axhline(instab_bound_5[0], linestyle="dotted")
    ax.axhline(instab_bound_1[0], linestyle="dotted")

    ax.set_xlabel(r"averaging time $T_{\mathrm{av}}$", fontsize=14)
    ax.set_ylabel(r"critical in-rate $\nu_{\mathrm{in}}^{\mathrm{crit}}$", fontsize=14)
    ax.set_yticks(np.arange(1, 1.4, 0.1))
    ax.set_xticks(np.arange(0, 60, 10))
    ax.set_yticklabels(np.array([1, 1.1, 1.2, 1.3]), fontsize=14)
    ax.set_xticklabels(np.arange(0, 60, 10), fontsize=14)
    ax.set_xlim((0, 50))
    ax.legend(loc="lower right")
