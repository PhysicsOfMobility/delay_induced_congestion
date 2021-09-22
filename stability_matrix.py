import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def varyf_periodicgrid(
    delay=0, f=0.1, numrep=100, nu_init=200, nu_final=256, dnu=2, tmax=400
):

    fname = (
        "periodicgrid_congestion_params_rep%i_tmax%i_r%i_%i_dr%i_f0_%i_tau%i.csv"
        % (numrep, tmax, nu_init, nu_final, dnu, 10 * f, delay)
    )
    output = pd.read_csv(fname)

    critbound_50 = 0
    critbound_25 = 0
    critbound_75 = 0

    current_nu = nu_init
    num_congested = 0
    for index in range(0, len(output["f"])):
        if output["r"][index] == current_nu:
            if output["congested"][index]:
                num_congested += 1
        else:
            if num_congested >= numrep / 4:
                if critbound_25 == 0:
                    critbound_25 = current_nu
            if num_congested >= numrep / 2:
                if critbound_50 == 0:
                    critbound_50 = current_nu
            if num_congested >= 3 * numrep / 4:
                if critbound_75 == 0:
                    critbound_75 = current_nu
            current_nu += dnu
            num_congested = 0

    return critbound_25, critbound_50, critbound_75


def critvalues_periodicgrid(tau=0):

    fvals = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    if tau == 0:
        low_bounds = [200, 210, 224, 236, 236, 230]
        high_bounds = [230, 240, 246, 250, 250, 256]
    if tau == 5:
        low_bounds = [196, 206, 214, 218, 210, 204]
        high_bounds = [222, 234, 234, 234, 234, 228]
    if tau == 15:
        low_bounds = [180, 204, 200, 190, 190, 184]
        high_bounds = [230, 230, 222, 216, 216, 202]

    critvals_25 = []
    critvals_50 = []
    critvals_75 = []

    for idx, f in enumerate(fvals):
        bound_25, bound_50, bound_75 = varyf_periodicgrid(
            f=f, nu_init=low_bounds[idx], nu_final=high_bounds[idx], delay=tau
        )
        critvals_25.append(bound_25)
        critvals_50.append(bound_50)
        critvals_75.append(bound_75)

    return critvals_25, critvals_50, critvals_75


def numruns(file):
    output = pd.read_csv(file)
    old_rate = 86
    old_delay = 15
    result_list = []
    for idx in range(0, len(output["r"])):
        current_rate = output["r"][idx]
        current_delay = output["delay"][idx]
        if current_rate == old_rate and current_delay == old_delay:
            if output["congested_rt"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)
    return result_list


def frac_congested(file):
    output = pd.read_csv(file)
    runtime_vals = np.arange(50, 1000, 50)
    results_array = np.zeros(len(runtime_vals))
    res_idx = 0
    for idx in range(0, len(output["simtime"])):
        if output["repetition"][idx] == 0:
            num_congested = 0
            if output["congested"][idx] == True:
                num_congested += 1
        elif output["repetition"][idx] > 0 and output["repetition"][idx] < 99:
            if output["congested"][idx] == True:
                num_congested += 1
        else:
            if output["congested"][idx] == True:
                num_congested += 1
            results_array[res_idx] = num_congested / 100
            res_idx += 1
    return results_array


def phasematrix(file):
    output = pd.read_csv(file)
    rs = np.arange(70, 122, 2)
    delays = np.arange(0, 21, 1)
    rep = 100
    stab_matrix = np.zeros((len(rs), len(delays)))
    old_rate = 0
    old_delay = 0
    result_list = []

    for idx in range(0, len(output["r"])):
        current_rate = output["r"][idx]
        current_delay = output["delay"][idx]

        if current_rate != old_rate or current_delay != old_delay:
            if len(result_list) > 0:
                num_congested = sum(result_list)
                ratio_congested = num_congested / rep
                r_idx = np.where(rs == old_rate)[0][0]
                del_idx = np.where(delays == old_delay)[0][0]
                stab_matrix[r_idx][del_idx] = ratio_congested

            result_list = []
            old_rate = current_rate
            old_delay = current_delay
            if output["congested"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)
        else:
            if output["congested"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)

    return stab_matrix


def stability_varyf(file, delay, rvals):
    output = pd.read_csv(file)
    rs = rvals
    fvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    delayvals = [delay]
    rep = 100
    stab_matrix = np.zeros((3, len(rs), len(fvals)))
    old_rate = 0
    old_f = 0
    old_delay = 0
    result_list = []

    for idx in range(0, len(output["r"])):
        current_delay = output["delay"][idx]
        current_f = output["f"][idx]
        current_rate = output["r"][idx]

        if current_rate != old_rate or current_f != old_f or current_delay != old_delay:
            if len(result_list) > 0:
                num_congested = sum(result_list)
                ratio_congested = num_congested / rep
                r_idx = np.where(rs == old_rate)[0][0]
                f_idx = np.where(fvals == old_f)[0][0]
                d_idx = np.where(delayvals == old_delay)
                stab_matrix[d_idx, r_idx, f_idx] = ratio_congested

            result_list = []
            old_rate = current_rate
            old_f = current_f
            old_delay = current_delay
            if output["congested"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)
        else:
            if output["congested"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)

    return stab_matrix


def critical_rates(stab_matrix, rvals, boundvals=[0.25, 0.5, 0.75], delay=15):
    rs = rvals
    fvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    delayvals = [delay]
    critval_dict = {}
    for bound in boundvals:
        crit_vals = np.zeros(len(fvals))
        for d_idx in range(0, len(delayvals)):
            for f_idx in range(0, len(fvals)):
                for r_idx in range(0, len(rs)):
                    if stab_matrix[d_idx, r_idx, f_idx] > bound:
                        crit_vals[f_idx] = rs[r_idx]
                        break
        critval_dict.update({bound: crit_vals})
    return critval_dict


def bound_line(stab_matrix, bound):
    rs = np.arange(70, 122, 2)
    delays = np.arange(0, 21, 1)
    crit_vals = np.zeros(len(delays))
    for d_idx in range(0, len(delays)):
        for r_idx in range(0, len(rs)):
            if stab_matrix[r_idx, d_idx] > bound:
                crit_vals[d_idx] = rs[r_idx]
                break
    return crit_vals


def plot_critvals(crit_vals_del1, crit_vals_del5, crit_vals_del15):
    fvals = np.arange(0, 1.1, 0.1)
    _, ax = plt.subplots()

    plt.plot(fvals, crit_vals_del1[0.25], "b--", label="delay 1, bound=0.25")
    plt.plot(fvals, crit_vals_del1[0.5], "bo-", label="delay 1, bound=0.5")
    plt.plot(fvals, crit_vals_del1[0.75], "b:", label="delay 1, bound=0.75")

    plt.plot(fvals, crit_vals_del5[0.25], "r--", label="delay 5, bound=0.25")
    plt.plot(fvals, crit_vals_del5[0.5], "rs-", label="delay 5, bound=0.5")
    plt.plot(fvals, crit_vals_del5[0.75], "r:", label="delay 5, bound=0.75")

    plt.plot(fvals, crit_vals_del15[0.25], "g--", label="delay 15, bound=0.25")
    plt.plot(fvals, crit_vals_del15[0.5], "g^-", label="delay 15, bound=0.5")
    plt.plot(fvals, crit_vals_del15[0.75], "g:", label="delay 15, bound=0.75")

    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(75, 115, 10))
    ax.set_xticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), fontsize=14)
    ax.set_yticklabels(np.arange(75, 115, 10), fontsize=14)
    ax.set_xlabel("fraction of informed drivers", fontsize=14)
    ax.set_ylabel("critical in-rate", fontsize=14)

    plt.legend()


def elongate_matrix(stab_matrix, file):
    output = pd.read_csv(file)

    rs = np.arange(70, 120, 2)
    delays = np.arange(21, 26, 1)
    rep = 100
    additional_matrix = np.zeros((len(rs), len(delays)))
    old_rate = 0
    old_delay = 0

    result_list = []
    for idx in range(0, len(output["r"])):
        current_rate = output["r"][idx]
        current_delay = output["delay"][idx]

        if current_rate != old_rate or current_delay != old_delay:
            if len(result_list) > 0:
                num_congested = sum(result_list)
                ratio_congested = num_congested / rep

                r_idx = np.where(rs == old_rate)[0][0]
                del_idx = np.where(delays == old_delay)[0][0]

                additional_matrix[r_idx][del_idx] = ratio_congested

            result_list = []
            old_rate = current_rate
            old_delay = current_delay
            if output["congested"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)
        else:
            if output["congested"][idx] == True:
                result_list.append(1)
            else:
                result_list.append(0)

    complete_matrix = np.concatenate((stab_matrix, additional_matrix), axis=1)
    return complete_matrix


def black_and_white(stab_matrix, boundary=0.5):
    bw_matrix = np.zeros((len(stab_matrix[:, 0]), len(stab_matrix[0, :])))
    for i in range(0, len(stab_matrix[:, 0])):
        for j in range(0, len(stab_matrix[0, :])):
            if stab_matrix[i, j] < boundary:
                bw_matrix[i, j] = 0
            else:
                bw_matrix[i, j] = 1
    return bw_matrix


def plot_phaseplot_avg(stab_matrix_avg):
    plt.imshow(
        stab_matrix_avg[:-2, :],
        cmap="viridis",
        origin="lower",
        extent=(0, 25, 70, 116),
        aspect="auto",
    )
    plt.colorbar()


def plot_phaseplot_noavg(stab_matrix_noavg, critline):
    _, ax = plt.subplots()
    plt.imshow(
        stab_matrix_noavg[:-2, :],
        cmap="viridis",
        origin="lower",
        extent=(0, 20, 70, 118),
        aspect="auto",
    )

    markers = [(0, 85), (5, 85), (15, 85)]
    x, y = zip(*markers)
    ax.plot(x, y, "o", c="red")

    # line which marks where half of simulations congest
    delays = np.arange(0, 21, 1)
    plt.plot(delays, critline, "--", color="darkgrey")
    plt.colorbar()
    ax.set_yticks(np.arange(70, 120, 10))
    ax.set_xticks(np.arange(0, 24, 5))
    ax.set_yticklabels(np.arange(70, 120, 10), fontsize=14)
    ax.set_xticklabels(np.arange(0, 24, 5), fontsize=14)
    ax.set_xlabel("delay", fontsize=14)
    ax.set_ylabel("in-rate", fontsize=14)


def plot_phaseplot_diff(
    stab_matrix_avg, stab_matrix_noavg, critline_avg, critline_noavg
):
    _, ax = plt.subplots()
    cmap = mpl.cm.get_cmap("bwr")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    plt.imshow(
        stab_matrix_avg[:-2, :] - stab_matrix_noavg[:-2, :],
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=(0, 20, 70, 118),
        aspect="auto",
    )
    plt.colorbar()
    ax.set_yticks(np.arange(70, 120, 10))
    ax.set_xticks(np.arange(0, 24, 5))
    ax.set_yticklabels(np.arange(70, 120, 10), fontsize=14)
    ax.set_xticklabels(np.arange(0, 24, 5), fontsize=14)

    # line which marks where half of simulations congest
    delays = np.arange(0, 21, 1)
    plt.plot(delays, critline_noavg, "--", color="darkgrey")
    plt.plot(delays, critline_avg, ".-", color="lightgrey")

    ax.set_xlabel("delay", fontsize=14)
    ax.set_ylabel("in-rate", fontsize=14)
