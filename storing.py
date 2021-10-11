import numpy as np


def record_state(env, resolution=1):
    """A simpy process to record the occupation numbers of all streets
    in regular intervals of size resolution
    
    Parameters
    ----------
    env : simpy simulation environment
    resolution : float, default 1
                interval time between measurements
    
    Returns
    --------
    """
    env.state_time_resolution = resolution
    streets = env.network.get_streets_flat()
    while True:
        yield env.timeout(resolution)
        current = np.zeros((1, len(streets)))
        for i, street in enumerate(streets):
            current[0, i] = env.network.graph[street[0]][street[1]]["numcars"]
        env.state = np.concatenate((env.state, current))

