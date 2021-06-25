import numpy as np


def avg_time(env):
    """calculate the average time a car needs for one segment, after the simulation has run,
    from the state vector. This is weighted by cars
    ! This assumes N_0 and t_0 is equal for all streets
    currently broken because non-existing streets have -1 entries
    """
    print(f"assuming N_0={env.N_0}, t_0={env.t_0} for all streets")
    N = env.state

    return np.sum(env.t_0 * env.N_0 * (np.exp(N / env.N_0) - 1), axis=1) / np.sum(N, axis=1)


def avg_expected_time(env, exclude_first=7):
    """calculates the average expected time per edge of cars starting at times t_start"""
    t_travel = []
    t_start = []
    for car in env.cars:
        s_t = car.starttime
        if (car.starttime - env.delay) > exclude_first and car.steps_traveled:
            p_t = car.expected_time / len(car.path)
            t_start.append(s_t)
            t_travel.append(p_t)

    return np.array(t_start), np.array(t_travel)


def avg_real_time(env, exclude_first=7):
    """calculates the average time per edge cars starting at t_start actually needed
    to travel the system. only draws time from cars that have reached the end
    of their path at the end of the simulation."""
    t_travel = []
    t_start = []
    for car in env.cars:
        if (car.starttime - env.delay) > exclude_first and car.steps_traveled:
            t_start.append(car.starttime)
            t_travel.append(car.real_time / car.steps_traveled)
    return np.array(t_start), np.array(t_travel)


def total_real_time(env, exclude_first=20):
    """calculate the average time per edge cars need, during the whole simulation, excluding those who started
    before time exclude_first."""
    _, t_travel = avg_real_time(env, exclude_first)
    return np.mean(t_travel)


def inner_outer(env):
    """Calculate distribution of traffic on inner and outer edges.
    Only the outermost edges (forming a rectangle around the system) are considered 'outer'
    """
    nx = env.network.n_x
    ny = env.network.n_y
    edge_low = list(range(0, 4 * nx, 4))
    edge_high = list(range(4 * (ny - 1) * nx, 4 * ny * nx, 4))
    edge_left = list(range(0, 4 * nx * ny, 4 * nx))
    edge_right = list(range(4 * (nx - 1), 4 * nx * ny, 4 * nx))
    outer_edges = []
    for i in range(nx - 1):
        outer_edges += [
            edge_low[i] + 3,
            edge_high[i] + 3,
            edge_low[nx - 1 - i] + 1,
            edge_high[nx - 1 - i] + 1,
        ]

    for i in range(ny - 1):
        outer_edges += [
            edge_left[i] + 0,
            edge_right[i] + 0,
            edge_left[ny - 1 - i] + 2,
            edge_right[ny - 1 - i] + 2,
        ]

    n_outer = len(outer_edges)
    n_total = np.count_nonzero(env.state[0] >= 0)
    n_inner = n_total - n_outer
    print(f"outer {n_outer}, inner {n_inner}")

    result = np.empty(
        (len(env.times), 3)
    )  # a np array in which the rows are the time steps and the columns are time, inner, outer
    result[:, 0] = env.times
    outer = np.sum(env.state[:, outer_edges], axis=1)
    # total number of cars on outer edges
    result[:, 1] = (np.sum(env.state * (env.state >= 0), axis=1) - outer) / n_inner
    # * (env.state >= 0) to filter out all the entries which are -1 because street doesn't exist
    result[:, 2] = outer / n_outer
    return result


def total_cars(env):
    """ calculate total # of cars in the system for all times """
    result = np.empty((len(env.times), 2))
    result[:, 0] = env.times

    result[:, 1] = np.sum(env.state * (env.state >= 0), axis=1)
    return result


def informed_drivers(env):
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


def is_congested(env, method="ratio"):
    if method == "diff":
        cars = total_cars(env)
        recent = cars[-20:, 1]
        diff = (recent - np.roll(recent, 1))[1:]
        if np.count_nonzero(diff > 0) > 0.8 * len(diff):
            return True
        else:
            return False
    elif method == "ratio":
        try:
            congested = env.congested
            if congested:
                return True
        except AttributeError:
            pass
        resolution = env.times[1] - env.times[0]
        lookback = int(25 / resolution)
        avg_over = int(10 / resolution)
        pm = int(5 / resolution)
        threshold = env.r + 3 * np.sqrt(env.r)
        cars = total_cars(env).T[1]
        now = np.mean(cars[-avg_over:])
        past = np.mean(cars[-lookback - pm : -lookback + pm])
        if now - past > 2 * threshold:
            print("congested according to ratio")
            return True
        else:
            return False
    elif method == "runtime":
        cars_on_roads = env.state[-1]
        crit = np.max(cars_on_roads)
        if crit > 100:
            env.congested = True
            print("congested at time", env.now)
            return True
        else:
            env.congested = False
            return False


def all_cars_streetwise(env):
    cars = np.sum(env.state, axis=0)
    return cars


def avg_cars_streetwise(info, data_type):
    if data_type == "environment":
        cars = np.mean(info.state[50:, :], axis=0)
    elif data_type == "statevector":
        cars = np.mean(info[50:, :], axis=0)

    return cars
