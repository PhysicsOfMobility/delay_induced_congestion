import numpy as np

import simulation


def repeat_sim(repetitions, envs=True, *args, **kwargs):
    """Repeat the same simulation a given number of times.
    Return a numpy array, which is all the state vectors stacked
    If envs is True, return a np array of envs instead
    *args and **kwargs are passed to do_sim
    """
    states = []
    for i in range(repetitions):
        if envs:
            states.append(simulation.do_sim(*args, **kwargs))
        else:
            states.append(simulation.do_sim(*args, **kwargs).state)

    return np.stack(states)


def vectorize(func):
    def vectorfunc(stackedstates):
        results = []
        for state in stackedstates:
            results.append(func(state))
        return np.stack(results)

    return vectorfunc
