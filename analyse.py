import numpy as np


def avg_time(env):
    """Calculate the average time a car needs for one segment at each measurement time,
    after the simulation has run, from the state vector. This is weighted by cars.
    ! This assumes N_0 and t_0 is equal for all streets
    
    Parameters
    ----------
    env : simpy simulation environment
    
    Returns
    -------
    array-like
    """
    print(f"assuming N_0={env.N_0}, t_0={env.t_0} for all streets")
    N = env.state

    return np.sum(env.t_0 * env.N_0 * (np.exp(N / env.N_0) - 1), axis=1) / np.sum(
        N, axis=1
    )


def avg_expected_time(env, exclude_first=5):
    """Calculate the average expected time per edge of cars starting at times t_start.
    Return array of start-times and array of expected times per edge.
    
    Parameters
    ----------
    env : simpy simulation environment
    exclude_first : int, default 5
                    when averaging, include only the elements of the state-vector starting from exclude_first+1
                    
    Returns
    -------
    array-like, array-like
    """
    t_travel = []
    t_start = []
    for car in env.cars:
        s_t = car.starttime
        if (car.starttime - env.delay) > exclude_first and car.steps_traveled:
            p_t = car.expected_time / len(car.path)
            t_start.append(s_t)
            t_travel.append(p_t)

    return np.array(t_start), np.array(t_travel)


def avg_real_time(env, exclude_first=5):
    """Calculate the average time per edge cars starting at t_start actually needed
    to travel the system.
    Only draw time from cars that have reached the end of their path
    at the end of the simulation.
    Return array of start-times and array of travel-times.
    
    Parameters
    ----------
    env : simpy simulation environment
    exclude_first : int, default 5
                    when averaging, include only the elements of the state-vector starting from exclude_first+1
    
    Returns
    -------
    array-like, array-like
    """
    t_travel = []
    t_start = []
    for car in env.cars:
        if (car.starttime - env.delay) > exclude_first and car.steps_traveled:
            t_start.append(car.starttime)
            t_travel.append(car.real_time / car.steps_traveled)
    return np.array(t_start), np.array(t_travel)


def total_real_time(env, exclude_first=5):
    """Calculate the average time per edge cars need, during the whole simulation,
    excluding those who started before time exclude_first.
    
    Parameters
    ----------
    env : simpy simulation environment
    exclude_first : int, default 5
                    when averaging, include only the elements of the state-vector starting from exclude_first+1
    
    Returns
    -------
    float
    """
    _, t_travel = avg_real_time(env, exclude_first)
    
    return np.mean(t_travel)


def total_cars(env):
    """Calculate total # of cars in the system for all measurement times.
    
    Parameters
    ----------
    env : simpy simulation environment
    
    Returns
    -------
    array-like
    """
    result = np.empty((len(env.times), 2))
    result[:, 0] = env.times

    result[:, 1] = np.sum(env.state * (env.state >= 0), axis=1)
    return result


def informed_drivers(env):
    """Return the fraction of informed drivers.
    
    Parameters
    ----------
    env : simpy simulation environment
    
    Returns
    -------
    float
    """
    num_informed = 0
    num_uninformed = 0

    for car in env.cars:
        if car.traffic_info == True:
            num_informed += 1
        else:
            num_uninformed += 1
    total_cars = num_informed + num_uninformed
    informed_part = num_informed / total_cars

    return informed_part


def is_congested(env, boundary=100):
    """Determine whether the network is congested.
    Return True if one street has more cars than the boundary value.
    
    Parameters
    ----------
    env : simpy simulation environment
    boundary : float, default 100
                critical number of cars
    
    Returns
    -------
    bool
    """

    cars_on_roads = env.state[-1]
    crit = np.max(cars_on_roads)
    if crit > boundary:
        env.congested = True
        print("congested at time", env.now)
        return True
    else:
        env.congested = False
        return False


def all_cars_streetwise(env):
    """Return the summed cars on each street.
    
    Parameters
    ----------
    env : simpy simulation environment
    
    Returns
    -------
    array-like
    """
    cars = np.sum(env.state, axis=0)
    return cars


def avg_cars_streetwise(info, data_type, exclude_first=5):
    """Return the average number of cars on each street from
    either env (data_type "environment")
    or from env.state (data_type "statevector").
    Only start measuring after exclude_first measurements
    
    Parameters
    ----------
    info : either simpy simulation environment or array-like
            provides information on street loads
    data_type : str
                provides information on the data fed in via info
                "environment" : info is simpy simulation environment
                "statevector" : info is the state-vector env.state
    exclude_first : int, default 5
                    when averaging, include only the elements of the state-vector starting from exclude_first+1
    
    Returns
    -------
    array-like
    """

    if data_type == "environment":
        cars = np.mean(info.state[exclude_first:, :], axis=0)
    elif data_type == "statevector":
        cars = np.mean(info[exclude_first:, :], axis=0)

    return cars
