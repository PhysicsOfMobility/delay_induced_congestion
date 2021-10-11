import numpy as np
import simpy as sp

import analyse
import storing_av
from cars import Car, DummyCar
from network_av import Network


def create_points(env):
    """Create random origin and destination nodes.
    
    Parameters
    ----------
    env : simpy simulation environment
    
    Returns
    -------
    int, int 
                origin, destination node
    """
    node_0 = np.random.randint(env.network.num_nodes)
    node_1 = np.random.randint(env.network.num_nodes)
    if node_0 == node_1:
        return create_points(env)
    else:
        return node_0, node_1


def car_creator(env, r, delay, f, beta):
    """Create new car objects

    Parameters
    ----------
    env : simpy simulation environment
    r : float
        avg rate of (cars/time unit) that spawn in the system
    delay : float
            information time delay
    f : float 
        fraction of informed drivers
    beta : float 
        parameter governing decision making in multinomial logit model
    
    Returns
    --------
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
    """For storing data about a simulation, to be able to pickle it"""

    def __init__(self, env: sp.Environment):
        """Initialize a DummyEnv environment
        
        Parameters
        ----------
        self : DummyEnv object
        env : simpy simulation environment
        
        Returns
        -------
        """
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


def do_sim(
    r=10,
    delay=5,
    t_0=1,
    N_0=10,
    beta=1,
    f=1,
    until=20,
    resolution=1,
    num_nodes=25,
    Tav=10,
    av_resolution=0.1,
    Ninit=0,
    periodic=True,
):
    """Run the simulation with given parameters.
    Return the simpy environment object which we use for storing everything about the simulation
    
    Parameters
    ----------
    r : float, default 85
        rate of incoming cars
    delay : float, default 15
            information time delay
    t_0 : float, default 1.0 
            time needed to travel an empty street
    N_0 : int, default 10
            street capacity
    beta : float, default 1.0
            parameter governing decision making in multinomial logit model
    f : float, default 1.0 
        fraction of informed drivers
    until : float, default 400.0 
            simulation duration
    resolution : float, default 1.0 
                time interval after which the simulation is recorded 
    num_nodes : int, default 25 
                number of nodes in the grid (has to be quadratic!)
    Tav : float, default 10
            averaging time window
    av_resolution : float, default 0.1
            interval between snapshots of street loads which are used for averaging
    Ninit : int, default 0 
            initial number of cars on all streets   
    periodic : bool, default True 
            determines whether the street network has periodic boundary conditions
            
    Returns 
    -------
    simpy simulation environment
    """
    env = sp.Environment()
    env.t_0 = t_0
    env.N_0 = N_0
    env.beta = beta
    env.delay = delay
    env.Tav = Tav
    env.av_resolution = av_resolution
    env.r = r
    env.f = f
    env.network = Network(env, num_nodes=num_nodes, t_0=t_0, N_0=N_0, periodic=periodic)
    env.numcars_dict = {}
    streets = env.network.get_streets_flat()
    # prepare the initial conditions for averaging
    for street in streets:
        env.numcars_dict.update({street: {"num_list": []}})
        max_listlength = env.Tav / env.av_resolution
        for _ in range(0, int(max_listlength)):
            env.numcars_dict[street]["num_list"].append(Ninit)
        env.numcars_dict[street].update(
            {"num_sum": sum(env.numcars_dict[street]["num_list"])}
        )
    env.state = np.empty((0, len(env.network.edges)))
    env.process(car_creator(env, r, delay, f, beta))
    env.process(storing_av.record_state(env, resolution=resolution))
    env.process(storing_av.update_list(env, resolution=av_resolution))

    t = 10  # seed simulation, to get data to check
    env.run(until=t)
    while (
        not analyse.is_congested(env) and t <= until
    ):  # while not congested and t smaller than end time
        t += 10
        env.run(until=t)

    env.times = np.arange(resolution, env.now, resolution)

    return env
