import numpy as np


class Car:
    def __init__(
        self,
        env,
        start: int,
        end: int,
        delay=0.0,
        traffic_info=True,
        beta=1.0,
    ):

        """Initiate a car object in the simulation environment.

        Parameters
        ----------
        self : Car object
        start : int
                origin of the driver (as a node index)
        end : int
            destination of the driver (as a node index)
        delay : float, default 0.0
                information delay
        traffic_info : bool, default True
            whether or not the driver decides based on signalled information 
        beta : float, default 1.0
            parameter governing decision making in multinomial logit model
            
        Returns
        -------
        """

        self.env = env
        self.start = start
        self.end = end
        # How old is the traffic information the cars base their choice on (if they use it)?
        self.delay = delay
        # Does the car use traffic information at all to choose their path?
        self.traffic_info = traffic_info
        self.beta = beta
        self.action = env.process(self.run())
        self.steps_traveled = 0
        self.real_time = 0
        self.expected_time = 0

    def run(self):
        """Choose a path from origin to destination, and travel on it.
        
        Parameters
        ----------
        self : Car object
        
        Returns 
        -------
        """
        paths = self.env.network.shortestpaths(
            self.start, self.end
        )  # possible paths to consider
        times = np.array([self.env.network.path_time(path) for path in paths])
        # Subtract a constant value from all times. This doesn't affect the
        # outcome because this results in a constant factor for the  probabilities
        # which will be normalised.
        # This helps avoid division by zero if p gets too small
        times_corrected = times - min(times)
        if self.traffic_info:
            p = np.exp(-1 * self.beta * times_corrected)
        else:
            p = np.ones((len(paths),))

        p = p / np.sum(p)
        path_index = np.random.choice(range(len(paths)), p=p)
        path = paths[path_index]
        self.expected_time = times[path_index]  # calculate time the car expects to need
        self.starttime = (
            self.env.now + self.delay
        )  # moved in front of delay, to catch errors when accessing the parameter
        self.path = path
        # now wait for a delay before departing
        yield self.env.timeout(self.delay)
        self.real_time = 0  # the actual time it will take to traverse the path
        for street in path:
            wait = self.env.network.graph[street[0]][street[1]]["traveltime"]
            self.real_time += wait
            self.steps_traveled += 1
            self.env.network.graph[street[0]][street[1]]["numcars"] += 1
            self.env.network.adjust_traveltime(street)
            yield self.env.timeout(wait)
            self.env.network.graph[street[0]][street[1]]["numcars"] -= 1
            self.env.network.adjust_traveltime(street)


class DummyCar:
    """A dummy class for storing the attributes associated with the car."""

    def __init__(self, car: Car):
        """
        Inititate a dummy car
        
        Parameters
        ----------
        self : DummyCar object
        car : Car object
        
        Returns
        --------
        
        """
        self.start, self.end, self.delay, self.traffic_info = (
            car.start,
            car.end,
            car.delay,
            car.traffic_info,
        )
        self.path = car.path
        self.expected_time, self.starttime, self.real_time = (
            car.expected_time,
            car.starttime,
            car.real_time,
        )
        self.steps_traveled = car.steps_traveled
