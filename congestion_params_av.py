import csv
import os
import pickle

import analyse
import simulation_av
import storing_av


def run_sims(
    until=400,
    rs=range(70, 122, 2),
    delays=range(0, 21, 1),
    Tav=50,
    fs=(1,),
    pointlist=None,
    repetitions=100,
    pickledir=None,
    outcomefn="congestion_params_av50_rep100_tmax400.csv",
):
    """Run simulations all combinations of r, delay from rs, delays. Write the envs as pickles in files in pickledir
    Write a summary as a csv into outcomefn
    """
    outcomes = []
    j = 0  # just to show progress

    # create dir if doesn't exist
    if pickledir is not None and (not os.path.isdir(pickledir)):
        os.mkdir(pickledir)

    if pointlist is None:
        pointlist = [(r, d, f) for r in rs for d in delays for f in fs]

    N = len(pointlist) * repetitions  # total number of simulations to be ran

    for point in pointlist:
        for i in range(repetitions):
            env = simulation_av.do_sim(r=point[0], delay=point[1], f=point[2], Tav=Tav, until=until + point[1])
            tttime = analyse.total_real_time(env)
            congested = analyse.is_congested(env)
            f_value = env.f
            outcomes.append([point[0], point[1], i, tttime, congested, f_value])
            dummyenv = simulation_av.DummyEnv(env)
            dummyenv.repetition = i
            if pickledir is not None:
                filename = os.path.join(pickledir, f"r{point[0]}delay{point[1]}rep{i}".replace(".", "_"))
                with open(filename, "wb") as picklefile:
                    pickle.dump(dummyenv, picklefile)

            j += 1
            perc = (j / N) * 100
            print(f"\rSimulation {j:>5}/{N} done ({perc:.1f}%)", end="")

    with open(outcomefn, "w") as file:
        writer = csv.writer(file)

        writer.writerow(["r", "delay", "repetition", "avgtime", "congested", "f"])
        writer.writerows(outcomes)


def compute_congested(pickledir, outcomefn):
    """ re-compute the outcomes-file for the simulation stored in pickledir """
    outcomes = []

    for fn in os.listdir(pickledir):
        path = os.path.join(pickledir, fn)
        env = storing_av.load_env(path)
        tttime = analyse.total_real_time(env)
        congested = analyse.is_congested(env)
        outcomes.append([env.r, env.delay, env.repetition, tttime, congested])

    with open(outcomefn, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["r", "delay", "repetition", "avgtime", "congested"])
        writer.writerows(outcomes)
