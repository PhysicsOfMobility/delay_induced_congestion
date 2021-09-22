import csv
import itertools
import os
import pickle
import numpy as np
import tempfile
import shutil
from joblib import Parallel, delayed

import analyse
import simulation
import simulation_av
import storing


def run_sims(
    periodic=True,
    until=400,
    rs=np.arange(180, 256, 2),
    delays=[0],
    fs=[0.1],
    pointlist=None,
    repetitions=100,
    pickledir=None,
    outcomefn="periodicgrid_congestion_params_rep100_tmax400_r180_256_dr2_f0_1_tau0.csv",
    n_jobs=-2,
    ):
    """Run simulations all combinations of (r, delay) from (rs, delays). 
    Write the simulation envs as pickles in files in pickledir.
    Write a summary as a csv into outcomefn.
    
    Keyword arguments:
    until -- maximal duration of simulation
    rs -- values of the in-rate parameter to simulate
    delays -- values of the delay parameter to simulate
    fs -- fractions of informed drivers
    pointlist -- parameter combinations for each simulation; defined below if None
    repetitions -- number of simulation runs per set of parameters
    pickledir -- directory to store simulation environments
    outcomefn -- filename of the output file
    n_jobs -- number of jobs for parallel computation
    """
    
    
    if pickledir is not None and (not os.path.isdir(pickledir)):
        os.mkdir(pickledir)

    if pointlist is None:
        pointlist = [(r, d, f) for r in rs for d in delays for f in fs]
    
    #Save the outcome of the parallel computations in a temporary directory.
    #This is necessary because apparently, global arrays are not shared between subprocesses 
    # in joblib. See https://stackoverflow.com/questions/34140560/accessing-and-altering-a-global-array-using-python-joblib 
    
    temppath = tempfile.mkdtemp()
    outcomespath = os.path.join(temppath, 'outcomes.mmap')
    columns = ["r", "delay", "repetition", "avgtime", "congested", "f", "informed"]
    outcomes = np.memmap(outcomespath, dtype = float, shape = (int(repetitions*len(pointlist)), len(columns)), mode = 'w+')
    
    # Parallel execution with joblib
    simlst = list(itertools.product(pointlist, range(repetitions)))
    print("Total number of simulation points:", len(simlst))
    

    def run_sim_point(point, i):
        """Joblib helper function for parallel execution"""
        env = simulation.do_sim(r=point[0], delay=point[1], f=point[2], until=until + point[1], periodic=periodic)
        tttime = analyse.total_real_time(env)
        congested = analyse.is_congested(env)
        f_value = env.f
        informed_part = analyse.informed_drivers(env)
        point_idx = int(repetitions*pointlist.index(point) + i)
        outcomes[point_idx, :] = np.array([point[0], point[1], i, tttime, congested, f_value, informed_part])
        dummyenv = simulation.DummyEnv(env)
        dummyenv.repetition = i
        if pickledir is not None:
            filename = os.path.join(pickledir, f"r{point[0]}delay{point[1]}rep{i}".replace(".", "_"))
            with open(filename, "wb") as picklefile:
                pickle.dump(dummyenv, picklefile)

    Parallel(n_jobs=n_jobs, verbose=100)(delayed(run_sim_point)(*args) for args in simlst)

    # Save results
    with open(outcomefn, "w") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(outcomes)

    # Delete the temporary directory and contents
    try:
        shutil.rmtree(temppath)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
    print("Done")
    
    
def run_sims_averaging(
    periodic=True,
    until=400,
    rs=np.arange(255, 260, 5),
    delays=range(0, 11),
    Tav=50,
    fs=(1,),
    pointlist=None,
    repetitions=10,
    pickledir=None,
    outcomefn="aveperiodicgrid_congestion_params_Tav50_rep10_tmax400.csv",
    n_jobs=-2
    ):
    
    """Run simulations with averaged information for all combinations of (r, delay) from (rs, delays). 
    Write the simulation envs as pickles in files in pickledir.
    Write a summary as a csv into outcomefn.
    
    Keyword arguments:
    until -- maximal duration of simulation
    rs -- values of the in-rate parameter to simulate
    delays -- values of the delay parameter to simulate
    Tav -- averaging time window (default 50)
    fs -- fractions of informed drivers
    pointlist -- parameter combinations for each simulation; defined below if None
    repetitions -- number of simulation runs per set of parameters (default 100)
    pickledir -- directory to store simulation environments
    outcomefn -- filename of the output file
    n_jobs -- number of jobs for parallel computation
    """
    
    # create dir if doesn't exist
    if pickledir is not None and (not os.path.isdir(pickledir)):
        os.mkdir(pickledir)

    if pointlist is None:
        pointlist = [(r, d, f) for r in rs for d in delays for f in fs]
    
    #Save the outcome of the parallel computations in a temporary directory.
    #This is necessary because apparently, global arrays are not shared between subprocesses 
    # in joblib. See https://stackoverflow.com/questions/34140560/accessing-and-altering-a-global-array-using-python-joblib 
    
    temppath = tempfile.mkdtemp()
    outcomespath = os.path.join(temppath, 'outcomes.mmap')
    columns = ["r", "delay", "repetition", "avgtime", "congested", "f", "informed"]
    outcomes = np.memmap(outcomespath, dtype = float, shape = (int(repetitions*len(pointlist)), len(columns)), mode = 'w+')
    
    # Parallel execution with joblib
    simlst = list(itertools.product(pointlist, range(repetitions)))
    print("Total number of simulation points:", len(simlst))

    
    def run_sim_point(point, i):
        """Joblib helper function for parallel execution"""
        env = simulation_av.do_sim(r=point[0], delay=point[1], f=point[2], Tav=Tav, until=until + point[1], periodic=periodic)
        tttime = analyse.total_real_time(env)
        congested = analyse.is_congested(env)
        f_value = env.f
        informed_part = analyse.informed_drivers(env)
        point_idx = int(repetitions*pointlist.index(point) + i)
        outcomes[point_idx, :] = np.array([point[0], point[1], i, tttime, congested, f_value, informed_part])
        dummyenv = simulation.DummyEnv(env)
        dummyenv.repetition = i
        if pickledir is not None:
            filename = os.path.join(pickledir, f"r{point[0]}delay{point[1]}rep{i}".replace(".", "_"))
            with open(filename, "wb") as picklefile:
                pickle.dump(dummyenv, picklefile)
    
    Parallel(n_jobs=n_jobs, verbose=100)(delayed(run_sim_point)(*args) for args in simlst)

    # Save results
    with open(outcomefn, "w") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(outcomes)

    # Delete the temporary directory and contents
    try:
        shutil.rmtree(temppath)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
    print("Done")


def compute_congested(pickledir, outcomefn):
    """ re-compute the outcomes-file for the simulation stored in pickledir """
    outcomes = []

    for fn in os.listdir(pickledir):
        path = os.path.join(pickledir, fn)
        env = storing.load_env(path)
        tttime = analyse.total_real_time(env)
        congested = analyse.is_congested(env)
        outcomes.append([env.r, env.delay, env.repetition, tttime, congested])

    with open(outcomefn, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["r", "delay", "repetition", "avgtime", "congested"])
        writer.writerows(outcomes)



run_sims(until=400,
    rs=np.arange(184, 202, 2),
    delays=[15],
    fs=[1],
    pointlist=None,
    repetitions=100,
    pickledir=None,
    outcomefn="periodicgrid_congestion_params_rep100_tmax400_r184_202_dr2_f0_10_tau15.csv",
    n_jobs=-2,)

