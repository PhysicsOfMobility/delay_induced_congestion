# -*- coding: utf-8 -*-

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def phaseplot(outcome_file, 
              critical_value=0.5,
              rs = np.arange(70, 122, 2),
              delays = np.arange(0, 21, 1),
              num_repetitions=100
              ):
    '''From simulation outcomes, generate a phase diagram to see for which parameters the street network congests.
    
    Parameters
    -----------
    outcome-file : str
                path to csv outcome file, generated using congestion_params.py
    critical_value : float, default 0.5
                    a delay-in-rate pair is considered critical if the fraction of congested simulations is > bound
    rs : array-like, default np.arange(70, 122, 2)
        in-rates
    delays : array-like, default np.arange(0, 21, 1) 
            delays
    rep : int, default 100 
            number of repetitions per paramater setting
            
    Returns 
    -------
    '''
    
    stability_matrix = phasematrix(file=outcome_file, 
                                   rs =rs, 
                                   delays = delays, 
                                   rep=num_repetitions,)
    
    critical_bound = bound_line(stab_matrix=stability_matrix, 
                                bound=critical_value,
                                rs=rs,
                                delays=delays)
    
    plot_phaseplot(stab_matrix=stability_matrix,
                         critline=critical_bound,
                         rs=rs, 
                         delays=delays, 
                         )
    

def phasediffplot(outcome_file_noavg,
                  outcome_file_avg,
                  critical_value=0.5,
                  rs = np.arange(70, 122, 2),
                  delays = np.arange(0, 21, 1),
                  num_repetitions=100):
    
    '''From simulation outcomes with and without averaging, generate a phase diagram 
    that shows whether averaging prevents congestion or not.
    
    Parameters
    -----------
    outcome_file_noavg : str
                        path to csv outcome file, generated using congestion_params.py without averaging
    outcome_file_avg : str 
                        path to csv outcome file, generated using congestion_params.py with averaging
    critical_value : float, default 0.5 
                    a delay is considered critical if the fraction of congested simulations is > bound
    rs : array-like, default np.arange(70, 122, 2) 
        in-rates
    delays : array-like, default np.arange(0, 21, 1) 
        delays
    rep : int, default 100 
        number of repetitions per paramater setting
        
    Returns 
    -------    
    '''
    
    stability_matrix_noavg = phasematrix(file=outcome_file_noavg, 
                                  rs =rs, 
                                  delays = delays, 
                                  rep=num_repetitions,)
   
    critical_bound_noavg = bound_line(stab_matrix=stability_matrix_noavg, 
                               bound=critical_value,
                               rs=rs,
                               delays=delays)
    
    stability_matrix_avg = phasematrix(file=outcome_file_avg, 
                                  rs =rs, 
                                  delays = delays, 
                                  rep=num_repetitions,)
   
    critical_bound_avg = bound_line(stab_matrix=stability_matrix_avg, 
                               bound=critical_value,
                               rs=rs,
                               delays=delays)
    
    plot_phaseplot_diff(stability_matrix_avg, 
                       stability_matrix_noavg, 
                       critical_bound_avg, 
                       critical_bound_noavg,
                       rs = np.arange(70, 122, 2),
                       delays = np.arange(0, 21, 1),)
    
def critrates_plot(parameter_dict = {1: {'outcome': 'data/file1.csv',
                                          'rs': np.arange(75, 120, 1)},
                                      5: {'outcome': 'data/file2.csv',
                                          'rs': np.arange(75, 120, 1)},
                                      15: {'outcome': 'data/file3.csv',
                                          'rs': np.arange(75, 120, 1)}},
                    fvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    rep=100,
                    boundvals = [0.25, 0.5, 0.75],
                    plotrange = np.arange(75, 120, 1)):
    '''From simulation outcomes, plot the critical in-rates for given fraction of informed drivers
    and delays. 
    
    Parameters
    ----------
    parameter_dict : dict
                    dictionary of dictionaries with 3 (!) delay values as keys, 
                    outcome filenames as one entry, and arrays of in-rates as a second entry
    fvals : list, default [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            fractions of informed drivers
    rep : int, default 100 
            number of simulations per parameter setting
    boundvals : list 
               3 critical ratios of congested simulations
    plotrange : array-like
                in-rates for the final plot
    
    Returns
    -------
    '''
    
    delays = list(parameter_dict.keys())
    
    criticalvalues_dict = {}
    
    for delay in delays:
        
        stability_matrix = stability_varyf(file=parameter_dict[delay]['outcome'], 
                                           delays = [delay], 
                                           rs=parameter_dict[delay]['rs'],
                                           fvals = fvals,
                                           rep=rep)
    
        critrates_dict = critical_rates(stab_matrix = stability_matrix,
                                        rs = parameter_dict[delay]['rs'],
                                        delays=[delay],
                                        boundvals = boundvals,
                                        fvals=fvals)
        
        criticalvalues_dict.update({delay: critrates_dict})
    
    plot_critvals(criticalvalues_dict[delays[0]],
                  criticalvalues_dict[delays[1]], 
                  criticalvalues_dict[delays[2]],
                  bounds=boundvals,
                  delays=delays,
                  fvals = fvals,
                  rs = plotrange)
    

def plot_critvalues_periodic(parameter_dict={0: {'low_bounds': [200, 210, 224, 236, 236, 230],
                                                 'high_bounds': [230, 240, 246, 250, 250, 256]},
                                             5: {'low_bounds': [196, 206, 214, 218, 210, 204],
                                                 'high_bounds': [222, 234, 234, 234, 234, 228]},
                                             15: {'low_bounds': [180, 204, 200, 190, 190, 184],
                                                  'high_bounds': [230, 230, 222, 216, 216, 202]}},
                             fvals = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
                             rep=100,
                             tmax=400,
                             boundvals=[0.25, 0.5, 0.75],
                             plotrange = np.arange(180, 260, 1)):
    '''For the periodic grid, evaluate results to find the critical in-rates for given delay 
    and fraction of informed drivers. Plot the critical in-rates against the fractions
    of informed drivers.
    
    Parameters
    ----------
    parameter_dict : dict
                    dictionary of dictionaries with 3 (!) delay values as keys, 
                    outcome filenames as one entry, and arrays of in-rates as a second entry
    fvals : list, default [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            fractions of informed drivers
    rep : int, default 100 
            number of simulations per parameter setting
    tmax : float, default 400
            duration of the given simulation
    boundvals : list 
               3 critical ratios of congested simulations
    plotrange : array-like
                in-rates for the final plot
    
    Returns
    -------
    '''
    
    
    delays=list(parameter_dict.keys())
    resultsdict = {}
    for delay in delays:
        
        critvals_0, critvals_1, critvals_2 = critvalues_periodicgrid(tau=delay,
                                                                        low_bounds=parameter_dict[delay]['low_bounds'], 
                                                                        high_bounds=parameter_dict[delay]['high_bounds'], 
                                                                        fvals=fvals,
                                                                        boundvals=boundvals)
        resultsdict.update({delay: {boundvals[0]: critvals_0,
                                    boundvals[1]: critvals_1,
                                    boundvals[2]: critvals_2}})
    
    plot_critvals(resultsdict[delays[0]],
                  resultsdict[delays[1]],
                  resultsdict[delays[2]],
                  bounds=boundvals,
                  delays=delays,
                  fvals=fvals,
                  rs=plotrange,)



def phasematrix(file,
                rs = np.arange(70, 122, 2),
                delays = np.arange(0, 21, 1),
                rep = 100
                ):
    '''From simulation outcomes, generate a numpy array that for each pair of in-rate
    and delay gives the fraction of simulations that end in a congested state.
    
    Parameters
    ----------
    file : str
            path to csv outcome file, generated using congestion_params.py
    rs : array-like, default np.arange(70, 122, 2) 
            in-rates 
    delays : array-like, default np.arange(0, 21, 1)
            delays
    rep : int, default 100 
            number of repetitions per paramater setting
            
    Returns 
    -------
    array-like
    '''
    
    stab_matrix = np.zeros((len(rs), len(delays)))
    old_rate = 0
    old_delay = 0
    result_list = []
    
    output = pd.read_csv(file)

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

def stability_varyf(file, 
                    delays = [10], 
                    rs=np.arange(75, 122, 1),
                    fvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    rep=100
                    ):
    '''Find the fraction of congested simulations given a delay, in-rate and fraction of informed drivers.
    
    Parameters
    ----------
    file : str
            path to csv-file containing simulation output, generated using congestion_params.py
    delays : list, default [10] 
            delays
    rs : array-like, default np.arange(75, 122, 1) 
        in-rates
    fvals : list, default [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
            fractions of informed drivers
    rep : int, default 100 
            number of simulations per set of parameters
            
    Returns
    --------
    array-like
    '''
    
    output = pd.read_csv(file)
    stab_matrix = np.zeros((len(delays), len(rs), len(fvals)))
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
                d_idx = np.where(delays == old_delay)
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

def bound_line(stab_matrix,
               bound=0.5,
               rs = np.arange(70, 122, 2),
               delays = np.arange(0, 21, 1),
               ):
    
    '''Return an array of the delays at which a transition from free-flow to congestion occurs.
    
    Parameters
    ----------
    stab_matrix : array-like
                    2-dimensional numpy array that gives the fraction of congested simulations
                    for each parameter pair.
    bound : float, default 0.5 
            a delay is considered critical if the fraction of congested simulations is > bound
    rs : array-like, default np.arange(70, 122, 2) 
        in-rates
    delays : array-like, default np.arange(0, 21, 1) 
            delays
            
    Returns 
    -------
    array-like
    '''
    
    crit_vals = np.zeros(len(delays))
    for d_idx in range(0, len(delays)):
        for r_idx in range(0, len(rs)):
            if stab_matrix[r_idx, d_idx] > bound:
                crit_vals[d_idx] = rs[r_idx]
                break
    return crit_vals

def critical_rates(stab_matrix, 
                   rs=np.arange(75, 120, 1), 
                   delays = [1],
                   boundvals=[0.25, 0.5, 0.75],
                   fvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    '''From a matrix that gives the fraction of congested simulation runs, find the critical in-rates.
    Return a dictionary with boundvals as keys and arrays of critical in-rates as values.
    
    Parameters
    -----------
    stab_matrix : array-like
                    3-dimensional numpy array that gives the fraction of congested simulations
                    for each set of in-rate, delay and fraction of informed drivers
    rs : array-like, default np.arange(75, 120, 1) 
        in-rates
    delays : list, default [1] 
        delays
    boundvals : list, default [0.25, 0.5, 0.75] 
                critical fractions of congested simulation runs for which the critical in-rates will be computed
    fvals : list, default [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
            fractions of informed drivers
    
    Returns 
    -------
    dict
            keys : critical fraction of congested simulation runs
            values : corresponding in-rates for each fraction of informed drivers
    '''
    
    critval_dict = {}
    for bound in boundvals:
        crit_vals = np.zeros(len(fvals))
        for d_idx in range(0, len(delays)):
            for f_idx in range(0, len(fvals)):
                for r_idx in range(0, len(rs)):
                    if stab_matrix[d_idx, r_idx, f_idx] > bound:
                        crit_vals[f_idx] = rs[r_idx]
                        break
        critval_dict.update({bound: crit_vals})
    return critval_dict


def plot_phaseplot(stab_matrix,
                         critline,
                         rs = np.arange(70, 122, 2),
                         delays = np.arange(0, 21, 1),
                         delay_markers = [0, 5, 15],
                         rate_marker = 85
                         ):
    '''Plot a phase diagram to see for which parameters the street network congests.
    
    Parameters
    -----------
    stab_matrix : array-like 
                    2-dimensional numpy array that gives the fraction of congested simulations
                    for each parameter pair
    critline : array-like
                delays at which a transition from free-flow to congestion occurs
    rs : array-like, default np.arange(70, 122, 2)
        in-rates
    delays : array-like, default np.arange(0, 21, 1)
            delays
    delay_markers : list, default [0, 5, 15] 
                    delay values that will be marked in the plot
    rate_marker : float, default 85 
                    in-rate at which delays will be marked in the plot
    Returns 
    -------
    '''
    
    
    _, ax = plt.subplots()
    plt.imshow(
        stab_matrix,
        cmap="viridis",
        origin="lower",
        extent=(delays[0], delays[-1], rs[0], rs[-1]),
        aspect="auto",
    )

    markers = [(delay_markers[i], rate_marker) for i in range(0, len(delay_markers))]
    x, y = zip(*markers)
    ax.plot(x, y, "o", c="red")

    # line which marks where half of simulations congest
    plt.plot(delays, critline, "--", color="darkgrey")
    plt.colorbar()
    ax.set_yticks(np.arange(rs[0], rs[-1]+10, 10))
    ax.set_xticks(np.arange(delays[0], delays[-1]+5, 5))
    ax.set_yticklabels(np.arange(rs[0], rs[-1]+10, 10), fontsize=14)
    ax.set_xticklabels(np.arange(delays[0], delays[-1]+5, 5), fontsize=14)
    ax.set_xlabel("delay", fontsize=14)
    ax.set_ylabel("in-rate", fontsize=14)
    

def plot_phaseplot_diff(stab_matrix_avg, 
                        stab_matrix_noavg, 
                        critline_avg, 
                        critline_noavg,
                        rs = np.arange(70, 122, 2),
                        delays = np.arange(0, 21, 1),
):
    '''Plot a phase diagram that shows whether averaging prevents congestion or not. 
    
    Parameters
    -----------
    stab_matrix_avg : array-like 
                    2-dimensional numpy array that gives the fraction of congested simulations
                    for each parameter pair with averaging.
    stab_matrix_noavg : array-like 
                    2-dimensional numpy array that gives the fraction of congested simulations
                    for each parameter pair without averaging.
    critline_avg : array-like 
                    delays at which a transition from free-flow to congestion occurs with averaging.
    critline_noavg : array-like 
                    array of delays at which a transition from free-flow to congestion occurs without averaging.
    rs : array-like, default np.arange(70, 122, 2) 
            in-rates
    delays : array-like, default np.arange(0, 21, 1) 
            delays
    '''
    
    _, ax = plt.subplots()
    cmap = mpl.cm.get_cmap("bwr")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    plt.imshow(
        stab_matrix_avg - stab_matrix_noavg,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=(delays[0], delays[-1], rs[0], rs[-1]),
        aspect="auto",
    )
    plt.colorbar()
    ax.set_yticks(np.arange(rs[0], rs[-1]+10, 10))
    ax.set_xticks(np.arange(delays[0], delays[-1]+5, 5))
    ax.set_yticklabels(np.arange(rs[0], rs[-1]+10, 10), fontsize=14)
    ax.set_xticklabels(np.arange(delays[0], delays[-1]+5, 5), fontsize=14)

    # line which marks where half of simulations congest
    plt.plot(delays, critline_noavg, "--", color="darkgrey")
    plt.plot(delays, critline_avg, ".-", color="lightgrey")

    ax.set_xlabel("delay", fontsize=14)
    ax.set_ylabel("in-rate", fontsize=14)

def plot_critvals(crit_vals_del1, 
                  crit_vals_del5, 
                  crit_vals_del15,
                  bounds=[0.25, 0.5, 0.75],
                  delays=[1,5,15],
                  fvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  rs = np.arange(75, 120, 1),
                  fval_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
                  ):
    '''Plot the critical in-rates corresponding to a given delay and fraction of informed drivers.
    
    Parameters
    -----------
    crit_vals_del1 : dict
                    dictionary with bounds as keys and arrays of critical in-rates as values; delay: delays[0]
    crit_vals_del5 : dict
                    dictionary with bounds as keys and arrays of critical in-rates as values; delay: delays[1]
    crit_vals_del15 : dict
                    dictionary with bounds as keys and arrays of critical in-rates as values; delay: delays[2]
    bounds : list, default [0.25, 0.5, 0.75]
            3 fractions of congested simulations at the given in-rates 
    delays : list, default [1,5,15] 
            3 delays that are considered here
    fvals : list, default [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            fractions of informed drivers
    rs : array-like, default np.arange(75, 120, 1)
        in-rates
    fval_ticks : list, default [0, 0.2, 0.4, 0.6, 0.8, 1]
        plot ticks for the fractions of informed drivers
    
    Returns 
    -------
    '''
        
    _, ax = plt.subplots()

    plt.plot(fvals, crit_vals_del1[bounds[0]], "b--", label="delay %i, bound=%.2f"%(delays[0], bounds[0]))
    plt.plot(fvals, crit_vals_del1[bounds[1]], "bo-", label="delay %i, bound=%.2f"%(delays[0], bounds[1]))
    plt.plot(fvals, crit_vals_del1[bounds[2]], "b:", label="delay %i, bound=%.2f"%(delays[0], bounds[2]))

    plt.plot(fvals, crit_vals_del5[bounds[0]], "r--", label="delay %i, bound=%.2f"%(delays[1], bounds[0]))
    plt.plot(fvals, crit_vals_del5[bounds[1]], "rs-",  label="delay %i, bound=%.2f"%(delays[1], bounds[1]))
    plt.plot(fvals, crit_vals_del5[bounds[2]], "r:", label="delay %i, bound=%.2f"%(delays[1], bounds[2]))

    plt.plot(fvals, crit_vals_del15[bounds[0]], "g--", label="delay %i, bound=%.2f"%(delays[2], bounds[0]))
    plt.plot(fvals, crit_vals_del15[bounds[1]], "g^-", label="delay %i, bound=%.2f"%(delays[2], bounds[1]))
    plt.plot(fvals, crit_vals_del15[bounds[2]], "g:", label="delay %i, bound=%.2f"%(delays[2], bounds[2]))

    ax.set_xticks(fval_ticks)
    ax.set_yticks(np.arange(rs[0], rs[-1]+10, 10))
    ax.set_xticklabels(fval_ticks, fontsize=14)
    ax.set_yticklabels(np.arange(rs[0], rs[-1]+10, 10), fontsize=14)
    ax.set_xlabel("fraction of informed drivers", fontsize=14)
    ax.set_ylabel("critical in-rate", fontsize=14)

    plt.legend()

def varyf_periodicgrid(
    delay=0, 
    f=0.1, 
    numrep=100, 
    nu_init=200, 
    nu_final=256, 
    dnu=2, 
    tmax=400,
    boundvals=[0.25, 0.5, 0.75]
):
    '''For one pair of delay and fraction of informed drivers in the periodic grid, get three critical in-rates.
    
    Parameters
    -----------
    delay : float, default 0 
            delay in the simulation
    f : float, default 0.1 
        fraction of informed drivers
    numrep : int, default 100
            number of simulations for the parameter setup
    nu_init : float, default 200 
            minimal in-rate
    nu_final : float, default 256 
            maximal in-rate
    dnu : float, default 2 
            stepsize between subsequent in-rates
    tmax : float, default 400 
            duration of a simulation
    boundvals : list, default [0.25, 0.5, 0.75] 
                3 critical fractions of congested simulations; 
                    !caution! the values have to be in ascending order
    
    Returns
    -------
    float, float, float
                        Critical in-rates corresponding to the boundvals
    '''
    

    fname = (
        "data/periodicgrid_congestion_params_rep%i_tmax%i_r%i_%i_dr%i_f0_%i_tau%i.csv"
        % (numrep, tmax, nu_init, nu_final, dnu, 10 * f, delay)
    )
    output = pd.read_csv(fname)

    critbound_0 = 0
    critbound_1 = 0
    critbound_2 = 0

    current_nu = nu_init
    num_congested = 0
    for index in range(0, len(output["f"])):
        if output["r"][index] == current_nu:
            if output["congested"][index]:
                num_congested += 1
        else:
            if num_congested >= boundvals[0]*numrep:
                if critbound_0 == 0:
                    critbound_0 = current_nu
            if num_congested >= boundvals[1]*numrep:
                if critbound_1 == 0:
                    critbound_1 = current_nu
            if num_congested >= boundvals[2]*numrep:
                if critbound_2 == 0:
                    critbound_2 = current_nu
            current_nu += dnu
            num_congested = 0

    return critbound_0, critbound_1, critbound_2


def critvalues_periodicgrid(tau, 
                            low_bounds, 
                            high_bounds,
                            fvals,
                            numrep=100, 
                            tmax=400,
                            boundvals=[0.25, 0.5, 0.75]):
    '''For a given delay, return the critical inrates for various fractions of informed drivers.
    
    Parameters
    ----------
    tau : float
            delay
    low_bounds : list 
                minimal in-rates for each fraction of informed drivers
    high_bounds : list 
                maximal in-rates for each fraction of informed drivers
    fvals : list
            fractions of informed drivers
    numrep : int, default 100 
            number of simulation runs per parameter setting
    tmax : float, default 400 
            duration of simulation
    boundvals : list, default [0.25, 0.5, 0.75] 
            3 critical fractions of congested simulations; 
                    !caution! the values have to be in ascending order
    
    Returns
    --------
    float, float, float
                        Critical in-rates corresponding to the boundvals
    '''
    
    critvals_0 = []
    critvals_1 = []
    critvals_2 = []

    for idx, f in enumerate(fvals):
        bound_0, bound_1, bound_2 = varyf_periodicgrid(
            f=f, 
            nu_init=low_bounds[idx],
            nu_final=high_bounds[idx],
            delay=tau,
            numrep=numrep,
            tmax=tmax
        )
        critvals_0.append(bound_0)
        critvals_1.append(bound_1)
        critvals_2.append(bound_2)

    return critvals_0, critvals_1, critvals_2





