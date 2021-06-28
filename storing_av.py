import csv
import pickle
import re

import numpy as np


def record_state(env, resolution=1):
    """a simpy process to record the occupation numbers of all streets
    in regular intervals of size resolution
    """
    env.state_time_resolution = resolution
    streets = env.network.get_streets_flat()
    while True:
        yield env.timeout(resolution)
        current = np.zeros((1, len(streets)))
        for i, street in enumerate(streets):
            if street is not None:
                current[0, i] = street.N
        env.state = np.concatenate((env.state, current))


def update_list(env, resolution=0.1):
    streets = env.network.get_streets_flat()
    while True:
        yield env.timeout(resolution)
        if env.now >= env.delay:
            list_lengths = []
            for i, street in enumerate(streets):
                if street is not None:
                    env.numcars_dict[street]["num_sum"] = (
                        env.numcars_dict[street]["num_sum"] - env.numcars_dict[street]["num_list"][0] + street.N
                    )
                    env.numcars_dict[street]["num_list"].pop(0)
                    env.numcars_dict[street]["num_list"].append(street.N)
                    list_lengths.append(len(env.numcars_dict[street]["num_list"]))


def store_state(env, file):
    """store the state of env in the file object
    not used anymore
    """
    writer = csv.writer(file)
    state_with_time = np.empty((env.state.shape[0], env.state.shape[1] + 1))
    state_with_time[:, 0] = env.times
    state_with_time[:, 1:] = env.state
    writer.writerows(state_with_time)


def load_state(filename, r, delay):
    """
    load the state of an env from filename
    The file consists of one or more csv parts, as outputted by store_state, separated by ---\nr:{r}, delay:{delay}
    Only choose the part where r and delay matches
    not used anymore
    """
    with open(filename) as file:
        filerows = iter(file)
        for line in filerows:
            if line == "----\n":
                argline = next(filerows)
                mr, mdelay = re.search(r"r=(\d+(?:\.\d+)?), delay=(\d+(?:\.\d+)?)", argline).groups()
                mr, mdelay = float(mr), float(mdelay)
                if r == mr and delay == mdelay:
                    break
        else:
            raise ValueError("No match found")

        rows = []
        reader = csv.reader(filerows)
        for row in reader:
            if row[0] == "----":  # end of the segment
                break
            rows.append(row)

    rows = np.asarray(rows, dtype=float)
    times = rows[:, 0]
    state = rows[:, 1:]

    class dummyEnv:
        pass

    env = dummyEnv()
    env.times = times
    env.state = state
    return env


def load_env(fn, r=None, delay=None, repetition=None):
    """Load a pickled environment from storage.
    If r, delay, repetition are not given, interpret fn as a filename
    If they are given, fn is the directory in which a file is to be found according to the naming in congestion_params.py
    """
    if r is not None:
        fn = fn + "/" + f"r{r}delay{delay}rep{repetition}".replace(".", "_")

    with open(fn, "rb") as file:
        env = pickle.load(file)

    return env