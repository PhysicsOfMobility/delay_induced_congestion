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


def update_list(env, resolution=0.1):
    """Update the stored history of street loads after a time resolution.
    For each street, generate a sum of the last street loads within the time window Tav.
    Add the current street load to the list of past loads, and delete the oldest one.
    
    Parameters
    ----------
    env : simpy simulation environment
    resolution : float, default 0.1
                interval time between updates of the street history
    
    Returns
    --------
    """
    streets = env.network.get_streets_flat()
    while True:
        yield env.timeout(resolution)
        if env.now >= env.delay:
            for i, street in enumerate(streets):
                env.numcars_dict[street]["num_sum"] = (
                    env.numcars_dict[street]["num_sum"]
                    - env.numcars_dict[street]["num_list"][0]
                    + env.network.graph[street[0]][street[1]]["numcars"]
                )
                env.numcars_dict[street]["num_list"].pop(0)
                env.numcars_dict[street]["num_list"].append(
                    env.network.graph[street[0]][street[1]]["numcars"]
                )


