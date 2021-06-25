import csv
import itertools
import os
import pickle

import numpy as np
from joblib import Parallel, delayed

import analyse
import simulation
import storing


def run_sims(
    until=400,
    rs=np.arange(70, 120, 2),
    delays=range(0, 21),
    fs=[1],
    pointlist=None,
    repetitions=100,
    pickledir=None,
    outcomefn="congestion_params_rep100_tmax400.csv",
    n_jobs=-2,
):
    """Run simulations all combinations of r, delay from rs, delays. Write the envs as pickles in files in pickledir
    Write a summary as a csv into outcomefn
    """
    outcomes = []

    if pickledir is not None and (not os.path.isdir(pickledir)):
        os.mkdir(pickledir)

    if pointlist is None:
        pointlist = [(r, d, f) for r in rs for d in delays for f in fs]

    # Parallel execution with joblib
    simlst = list(itertools.product(pointlist, range(repetitions)))
    print("Total number of simulation points:", len(simlst))

    def run_sim_point(point, i):
        """Joblib helper function for parallel execution"""
        env = simulation.do_sim(r=point[0], delay=point[1], f=point[2], until=until + point[1])
        tttime = analyse.total_real_time(env)
        congested = analyse.is_congested(env, "runtime")
        f_value = env.f
        informed_part = analyse.informed_drivers(env)
        outcomes.append([point[0], point[1], i, tttime, congested, f_value, informed_part])
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
        writer.writerow(["r", "delay", "repetition", "avgtime", "congested", "f", "informed"])
        writer.writerows(outcomes)

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
