import numpy as np
import simpy as sp

import analyse
import storing
from cars import Car, DummyCar
from network import Network


def create_points(env):
    """creates random start points for the cars to drive between"""
    x_0 = np.random.randint(env.network.n_x)
    y_0 = np.random.randint(env.network.n_y)
    x_1 = np.random.randint(env.network.n_x)
    y_1 = np.random.randint(env.network.n_y)
    if x_0 == x_1 and y_0 == y_1:
        return create_points(env)
    else:
        return (x_0, y_0), (x_1, y_1)


def car_creator(env, r, delay, f, beta):
    """
    r: avg rate of cars/time unit that spawn in the system
    delay: delay, or 'age' of the data the driver bases their information on
    beta: the beta param. for the probability distribution to choose the path
    f: the proportion of cars that rely on traffic information to make their decision.
    """
    env.cars = []
    while True:
        dt = np.random.exponential(1 / r)
        yield env.timeout(dt)
        start, end = create_points(env)
        # traffic_info = np.random.random() < f
        traffic_info = np.random.choice(np.array([0, 1]), p=np.array([1 - f, f]))
        if traffic_info == 1:
            traffic_info = True
        else:
            traffic_info = False
        env.cars.append((Car(env, start, end, delay, traffic_info, beta)))
        # print(f"car created at {env.now} start: {start}, end: {end}")


class DummyEnv:
    """ For storing data about a simulation, to be able to pickle it """

    def __init__(self, env: sp.Environment):
        self.t_0, self.N_0, self.delay, self.r, self.f = (
            env.t_0,
            env.N_0,
            env.delay,
            env.r,
            env.f,
        )
        self.state = env.state
        self.times = env.times
        self.cars = [DummyCar(car) for car in env.cars]


def do_sim(t_0=1, N_0=10, beta=1, r=85, delay=15, f=1, until=400, resolution=1):
    """Run the simulation with given parametres.
    Return the simpy environment object which we use for storing everything about the simulation
    """
    env = sp.Environment()
    env.t_0 = t_0
    env.N_0 = N_0
    env.beta = beta
    env.delay = delay
    env.r = r
    env.f = f
    env.network = Network(streetargs={"t_0": t_0, "N_0": N_0})
    env.state = np.empty((0, env.network.max_id()))
    env.process(car_creator(env, r, delay, f, beta))
    env.process(storing.record_state(env, resolution=resolution))

    t = 10  # seed simulation, to get data to check
    env.run(until=t)
    while not analyse.is_congested(env, "runtime") and t <= until:  # while not congested and t smaller than end time
        t += 10
        env.run(until=t)

    env.state = (
        env.state - 1
    )  # to correct the fact that we always counted the number of cars + 1 on each road for simplicity
    env.times = np.arange(resolution, env.now, resolution)

    return env
