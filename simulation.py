import numpy as np
import simpy as sp

import analyse
import storing
from cars import Car, DummyCar
from network import Network


def create_points(env):
    """creates random origin and destination nodes."""
    node_0 = np.random.randint(env.network.num_nodes)
    node_1 = np.random.randint(env.network.num_nodes)
    if node_0 == node_1:
        return create_points(env)
    else:
        return node_0, node_1


def car_creator(env, r, delay, f, beta):
    """Create new car objects
    
    Keyword arguments:
    env -- simulation environment
    r -- avg rate of (cars/time unit) that spawn in the system
    delay -- information time delay
    f -- fraction of informed drivers
    beta -- parameter governing decision making in multinomial logit model
    """
    env.cars = []
    while True:
        dt = np.random.exponential(1 / r)
        yield env.timeout(dt)
        start, end = create_points(env)
        traffic_info = np.random.choice(np.array([0, 1]), p=np.array([1 - f, f]))
        if traffic_info == 1:
            traffic_info = True
        else:
            traffic_info = False
        env.cars.append((Car(env, start, end, delay, traffic_info, beta)))
        


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


def do_sim(r=85, delay=15, t_0=1.0, N_0=10, beta=1.0, f=1.0, until=400.0, resolution=1.0):
    """Run the simulation with given parameters.
    Return the simpy environment object which we use for storing everything about the simulation
    
    Keyword arguments:
    r -- rate of incoming cars
    delay -- information time delay
    t_0 -- time needed to travel an empty street (default 1.0)
    N_0 -- street capacity (default 10)
    beta -- parameter governing decision making in multinomial logit model (default 1.0)
    f -- fraction of informed drivers (default 1.0)
    until -- simulation duration (default 400.0)
    resolution -- time interval after which the simulation is recorded (default 1.0)
    """
    env = sp.Environment()
    env.t_0 = t_0
    env.N_0 = N_0
    env.beta = beta
    env.delay = delay
    env.r = r
    env.f = f
    env.network = Network(num_nodes = 25, t_0 = t_0, N_0 = N_0)
    env.state = np.empty((0, len(env.network.edges)))
    env.process(car_creator(env, r, delay, f, beta))
    env.process(storing.record_state(env, resolution=resolution))

    t = 10  # seed simulation, to get data to check
    env.run(until=t)
    while not analyse.is_congested(env) and t <= until:  # while not congested and t smaller than end time
        t += 10
        env.run(until=t)
        
    env.times = np.arange(resolution, env.now, resolution)

    return env
