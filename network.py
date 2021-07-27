from functools import lru_cache
from typing import List, Tuple
import networkx as nx
import numpy as np


#class Street:
#    """ Street objects, on which the cars move."""
#    def __init__(self, t_0, N_0, id=0):
#        """ Construct a street with ID id, empty travel time t_0 and capacity N_0. """
#        self.N = 1  # one more than the actual number of cars on street
#        self.id = id
#        self.t_0 = t_0
#        self.N_0 = N_0
#
#    def t(self):
#        """Compute travel time given the current number of cars on the street."""
#        return self.t_0 * self.N_0 * (np.exp(self.N / self.N_0) - 1) / self.N


class Network:
    """Streets are edges of a directed networkx graph.
    """
    
    def __init__(self, num_nodes = 25, t_0 = 1, N_0 = 10):
        """Construct a street network, which is a networkx periodic directed grid. 
        
        Keyword arguments:
        num_nodes -- total number of nodes (has to be quadratic!)
        t_0 -- travel time of empty street
        N_0 -- effective capacity measure of streets
        """
        
        gridsize = int(np.sqrt(num_nodes))
        grid=nx.generators.grid_2d_graph(gridsize, gridsize)
        grid=nx.convert_node_labels_to_integers(grid)
        edgelist_grid = list(grid.edges()).copy()
        
        # Copy edges from the undirected graph to a directed graphs. 
        # Add edges which go in the opposite direction.
        dir_periodic_grid = nx.DiGraph()
        for edge in edgelist_grid:
            dir_periodic_grid.add_edge(edge[0], edge[1], 
                                       {'t_0': t_0, 'N_0': N_0, 
                                        'numcars': 0, 'traveltime': t_0})
            dir_periodic_grid.add_edge(edge[1], edge[0], 
                                       {'t_0': t_0, 'N_0': N_0, 
                                        'numcars': 0, 'traveltime': t_0})
    
        # Add edges connecting the sides and top and bottom, to ensure periodicity.
        for node in list(dir_periodic_grid.nodes()):
            if dir_periodic_grid.out_degree(node) < 4:
                if node % gridsize == 0:
                    dir_periodic_grid.add_edge(node, node+gridsize-1, 
                                               {'t_0': t_0, 'N_0': N_0, 
                                                'numcars': 0, 'traveltime': t_0})
                if (node+1) % gridsize == 0:
                    dir_periodic_grid.add_edge(node, node-gridsize+1, 
                                               {'t_0': t_0, 'N_0': N_0, 
                                                'numcars': 0, 'traveltime': t_0})
                if node-gridsize < 0:
                    dir_periodic_grid.add_edge(node, node + (gridsize-1)*gridsize, 
                                               {'t_0': t_0, 'N_0': N_0, 
                                                'numcars': 0, 'traveltime': t_0})
                if node+gridsize > 0:
                    dir_periodic_grid.add_edge(node, node - (gridsize-1)*gridsize, 
                                               {'t_0': t_0, 'N_0': N_0, 
                                                'numcars': 0, 'traveltime': t_0})
        
        self.graph = dir_periodic_grid    
        self.edges = list(dir_periodic_grid.edges())
        self.n_x = gridsize
        self.n_y = gridsize
        
    def get_streets_flat(self):
        """Get a list of the tuples which define streets in the network."""
        
        return self.edges
    
    def adjust_traveltime(self, edge):
        """Adjust the edge attribute 'traveltime' according to the current street load."""
        
        num_cars = self.graph[edge[0]][edge[1]]['numcars']
        t_0 = self.graph[edge[0]][edge[1]]['t_0']
        N_0 = self.graph[edge[0]][edge[1]]['N_0']
        
        if num_cars == 0:
            self.graph[edge[0]][edge[1]]['traveltime'] = t_0
        else:
            traveltime = t_0 * N_0 * (np.exp(num_cars / N_0) - 1) / num_cars
            self.graph[edge[0]][edge[1]]['traveltime'] = traveltime
    
    def get_traveltime(self, edge):
        """Get the current edge traveltime."""
        
        return self.graph[edge[0]][edge[1]]['traveltime']
    
    def shortestpaths(self, start, end, edgeweight = 't_0'):
        """Return a list of all shortest edge paths from start to end.
        
        Keyword arguments:
        start -- origin node
        end -- destination node
        edgeweight -- edge attribute used to measure edge length (default = 't_0')
        """
        graph = self.graph
        shortest_nodepaths = list(nx.all_shortest_paths(graph, start, end, 
                              weight=edgeweight, method='dijkstra'))
        shortest_paths = []
        for path in shortest_nodepaths:
            edgepath = []
            for i in range(len(path)-1):
                edgepath.append((path[i], path[i+1]))
            shortest_paths.append(edgepath)
        
        return shortest_paths
    
    def path_time(self, path):
        """Total time needed to traverse a path of streets."""
        
        time = 0
        for edge in path:
            time += self.graph[edge[0]][edge[1]]['traveltime']
            
        return time
    
#    def __init__(self, n_x=5, n_y=5, streetargs={"t_0": 1, "N_0": 10}):
#        """Construct a street network. 
#        
#        Keyword arguments:
#        n_x -- number of nodes in horizontal axis
#        n_y -- number of nodes in vertical axis
#        streetargs -- dictionary of street parameters t_0 and N_0
#        """
#        
#        self.n_x = n_x
#        self.n_y = n_y
#        # special... corner cases...
#        bottom_left = [Street(**streetargs), None, None, Street(**streetargs)]
#        bottom_right = [Street(**streetargs), Street(**streetargs), None, None]
#        top_right = [None, Street(**streetargs), Street(**streetargs), None]
#        top_left = [None, None, Street(**streetargs), Street(**streetargs)]
#
#        # bottom row
#        bottom_row = [bottom_left]
#        for x in range(1, n_x - 1):
#            node = []
#            for i in [1, 1, 0, 1]:
#                if i:
#                    node.append(Street(**streetargs))
#                else:
#                    node.append((None))
#            bottom_row.append(node)
#        bottom_row.append(bottom_right)
#
#        # center rows
#        center_rows = []
#        for y in range(1, n_y - 1):
#            # create each row:
#            center_row = []
#            # create start node
#            node = []
#            for i in [1, 0, 1, 1]:
#                if i:
#                    node.append(Street(**streetargs))
#                else:
#                    node.append((None))
#            center_row.append(node)
#
#            # create center nodes of each row
#            for x in range(1, n_x - 1):
#                node = []
#                for i in [1, 1, 1, 1]:
#                    node.append(Street(**streetargs))
#                center_row.append(node)
#
#            # create end node
#            node = []
#            for i in [1, 1, 1, 0]:
#                if i:
#                    node.append(Street(**streetargs))
#                else:
#                    node.append((None))
#            center_row.append(node)
#
#            center_rows += center_row
#
#        # top row
#        top_row = [top_left]
#        for x in range(1, n_x - 1):
#            node = []
#            for i in [0, 1, 1, 1]:
#                if i:
#                    node.append(Street(**streetargs))
#                else:
#                    node.append((None))
#            top_row.append(node)
#        top_row.append(top_right)
#
#        # List of all nodes from bottom left to top right, row-wise.
#        # Each node contains a list of street objects: [up, left, down, right]
#        # None if no street is attached.
#        self.streets = bottom_row + center_rows + top_row
#        for i, x in enumerate(self.streets):
#            for j, y in enumerate(x):
#                if y:
#                    y.id = 4 * i + j

#    def get_streets_flat(self):
#        """Get a flat list of all streets in the order in which they
#        appear in streets. The list is such that network.get_streets_flat()[street.id] = street
#        It contains None entries.
#        """
#        streets = []
#        for row in self.streets:
#            for street in row:
#                streets.append(street)
#        return streets
        
    

#    def get_street(self, start, end, get_id=False):
#        index = self.n_y * start[1] + start[0]
#        dx = end[0] - start[0]
#        dy = end[1] - start[1]
#        if dy == 1 and dx == 0:  # up direction index 0
#            direction = 0
#        elif dy == -1 and dx == 0:  # down direction index 2
#            direction = 2
#        elif dx == 1 and dy == 0:  # right direction index 3
#            direction = 3
#        elif dx == -1 and dy == 0:  # left direction index 1
#            direction = 1
#        try:
#            street = self.streets[index][direction]
#        except NameError:
#            raise ValueError("The nodes you put in are not adjacent.")
#
#        if street:
#            if get_id:
#                return street.id
#            else:
#                return street
#        else:
#            raise ValueError("The connection leads out of the city")
#
#    @lru_cache(maxsize=1024)
#    def shortestpaths(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Street]:
#        """ calculate all the shortest paths in a NxN grid from point start (x,y) to end (x,y) """
#        xdir = np.sign(end[0] - start[0])
#        ydir = np.sign(end[1] - start[1])
#
#        paths = [[start]]  # list of paths, with one segment being added each step
#        L = abs(end[1] - start[1]) + abs(end[0] - start[0])  # length of all the paths, in number of segments
#        for d in range(L):
#            nextlevel = []  # list of new paths
#            for path in paths:
#                lastpoint = path[-1]
#                if end[0] - lastpoint[0] != 0:  # if we can still continue into x-direction
#                    nextlevel.append(path + [(lastpoint[0] + xdir, lastpoint[1])])
#                if end[1] - lastpoint[1] != 0:  # if we can still continue into y-direction
#                    nextlevel.append(path + [(lastpoint[0], lastpoint[1] + ydir)])
#            paths = nextlevel
#
#        # convert to list of Streets instead of list of Points
#        streetpaths = []
#        for path in paths:
#            newpath = []
#            for i in range(len(path) - 1):
#                newpath.append(self.get_street(path[i], path[i + 1]))
#            streetpaths.append(newpath)
#        return streetpaths

#    def path_time(self, path):
#        """Total time needed to traverse a path of streets."""
#        time = 0
#        for street in path:
#            time += street.t()
#        return time

#    def max_id(self):
#        """ The maximal possible street id in the network."""
#        return self.n_x * self.n_y * 4
#
#    def get_coords_from_id(self, id):
#        """ From the street id, return the (x,y) positions of attached nodes."""
#        if id >= self.max_id():
#            raise IndexError("The input id is greater than the maximum amount of streets")
#        node_id = id // 4
#        direction = id % 4
#
#        x_0 = node_id % self.n_x
#        y_0 = node_id // self.n_x
#        if direction == 0:
#            y_1 = y_0 + 1
#            x_1 = x_0
#        elif direction == 1:
#            x_1 = x_0 - 1
#            y_1 = y_0
#        elif direction == 2:
#            y_1 = y_0 - 1
#            x_1 = x_0
#        elif direction == 3:
#            x_1 = x_0 + 1
#            y_1 = y_0
#
#        if 0 <= x_1 < self.n_x and 0 <= y_1 < self.n_y:
#            return (x_0, y_0), (x_1, y_1)
#        else:
#            raise ValueError("The requested street does not exist.")
#
#    def get_point_from_pointid(self, id):
#        """From the point id, return the (x,y) position of the node."""
#        x_0 = id % self.n_x
#        y_0 = id // self.n_x
#        return np.array([x_0, y_0])
