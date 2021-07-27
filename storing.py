import pickle
import numpy as np


def record_state(env, resolution=1):
    """A simpy process to record the occupation numbers of all streets
    in regular intervals of size resolution
    """
    env.state_time_resolution = resolution
    streets = env.network.get_streets_flat()
    while True:
        yield env.timeout(resolution)
        current = np.zeros((1, len(streets)))
        for i, street in enumerate(streets):
            current[0, i] = env.network.graph[street[0]][street[1]]['numcars']
        env.state = np.concatenate((env.state, current))


def load_env(fn, r=None, delay=None, repetition=None):
    """Load a pickled environment from storage and return the environment.
    If r, delay, repetition are not given, interpret fn as a filename
    If they are given, fn is the directory in which a file is to be found according to the naming in congestion_params.py
    """
    if r is not None:
        fn = fn + "/" + f"r{r}delay{delay}rep{repetition}".replace(".", "_")

    with open(fn, "rb") as file:
        env = pickle.load(file)

    return env
