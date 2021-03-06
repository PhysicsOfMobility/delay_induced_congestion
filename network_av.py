import networkx as nx
import numpy as np


class Network:
    """Streets are edges of a directed networkx graph."""

    def __init__(self, env, num_nodes=25, t_0=1, N_0=10, periodic=True):
        """Construct a street network, which is a networkx periodic directed grid.

        Parameters
        ----------
        self : Network object
        env : simpy simulation environment
        num_nodes : int, default 25 
                    total number of nodes (has to be quadratic!)
        t_0 : float, default 1 
                travel time of empty street
        N_0 : int, default 10 
                effective capacity measure of streets
        periodic : bool, default True
                    determines whether the street network has periodic boundary conditions
        
        Returns
        -------
        """

        gridsize = int(np.sqrt(num_nodes))
        grid = nx.generators.grid_2d_graph(gridsize, gridsize)
        grid = nx.convert_node_labels_to_integers(grid)
        edgelist_grid = list(grid.edges()).copy()

        # Copy edges from the undirected graph to a directed graphs.
        # Add edges which go in the opposite direction.
        dir_periodic_grid = nx.DiGraph()
        for edge in edgelist_grid:
            dir_periodic_grid.add_edge(
                edge[0], edge[1], t_0=t_0, N_0=N_0, numcars=0, traveltime=t_0
            )
            dir_periodic_grid.add_edge(
                edge[1], edge[0], t_0=t_0, N_0=N_0, numcars=0, traveltime=t_0
            )

        if periodic:
            # Add edges connecting the sides and top and bottom, to ensure periodicity.
            for node in list(dir_periodic_grid.nodes()):
                if dir_periodic_grid.out_degree(node) < 4:
                    if node % gridsize == 0:
                        dir_periodic_grid.add_edge(
                            node,
                            node + gridsize - 1,
                            t_0=t_0,
                            N_0=N_0,
                            numcars=0,
                            traveltime=t_0,
                        )
                    if (node + 1) % gridsize == 0:
                        dir_periodic_grid.add_edge(
                            node,
                            node - gridsize + 1,
                            t_0=t_0,
                            N_0=N_0,
                            numcars=0,
                            traveltime=t_0,
                        )
                    if node - gridsize < 0:
                        dir_periodic_grid.add_edge(
                            node,
                            node + (gridsize - 1) * gridsize,
                            t_0=t_0,
                            N_0=N_0,
                            numcars=0,
                            traveltime=t_0,
                        )
                    if node + gridsize >= num_nodes:
                        dir_periodic_grid.add_edge(
                            node,
                            node - (gridsize - 1) * gridsize,
                            t_0=t_0,
                            N_0=N_0,
                            numcars=0,
                            traveltime=t_0,
                        )

        self.graph = dir_periodic_grid
        self.edges = list(dir_periodic_grid.edges())
        self.n_x = gridsize
        self.n_y = gridsize
        self.num_nodes = num_nodes
        self.env = env

    def get_streets_flat(self):
        """Get a list of the tuples which define streets in the network.
        
        Parameters
        ----------
        self : Network object
        
        Returns
        -------
        list
        """
        
        return self.edges

    def adjust_traveltime(self, edge):
        """Adjust the edge attribute 'traveltime' according to the current street load.
        
        Parameters
        ----------
        self : Network object
        edge : tuple
                tuple of start-node and end-node, defining current edge
        
        Returns
        -------
        """

        num_cars = self.graph[edge[0]][edge[1]]["numcars"]
        t_0 = self.graph[edge[0]][edge[1]]["t_0"]
        N_0 = self.graph[edge[0]][edge[1]]["N_0"]

        if num_cars == 0:
            self.graph[edge[0]][edge[1]]["traveltime"] = t_0
        else:
            traveltime = t_0 * N_0 * (np.exp(num_cars / N_0) - 1) / num_cars
            self.graph[edge[0]][edge[1]]["traveltime"] = traveltime

    def get_traveltime(self, edge):
        """Get the current edge traveltime.
        
        Parameters
        ----------
        self : Network object
        edge : tuple
                tuple of start-node and end-node, defining current edge
        
        Returns
        -------
        float
        """

        return self.graph[edge[0]][edge[1]]["traveltime"]

    def shortestpaths(self, start, end, edgeweight="t_0"):
        """Return a list of all shortest edge paths from start to end.

        Parameters
        -----------
        self : Network object
        start : int
                origin node
        end : int
                destination node
        edgeweight : str, default 't_0'
                    edge attribute used to measure edge length
                    
        Returns
        -------
        list of lists of tuples
        """
        graph = self.graph
        shortest_nodepaths = list(
            nx.all_shortest_paths(
                graph, start, end, weight=edgeweight, method="dijkstra"
            )
        )
        shortest_paths = []
        for path in shortest_nodepaths:
            edgepath = []
            for i in range(len(path) - 1):
                edgepath.append((path[i], path[i + 1]))
            shortest_paths.append(edgepath)

        return shortest_paths

    def path_time(self, path):
        """Total time needed to traverse a path of streets.
        
        Parameters
        ----------
        self : Network object
        path : list of tuples
                a list of subsequently visited edges
        
        Returns
        -------
        float
        """
        time = 0
        listlength = self.env.Tav / self.env.av_resolution
        for edge in path:
            t_0 = self.graph[edge[0]][edge[1]]["t_0"]
            N_0 = self.graph[edge[0]][edge[1]]["N_0"]
            Nav = self.env.numcars_dict[edge]["num_sum"] / listlength
            if Nav == 0:
                time += t_0
            else:
                time += t_0 * N_0 * (np.exp(Nav / N_0) - 1) / Nav
        return time

    def node_positions(self, edge_length=1):
        """Find x and y coordinates of all nodes in the grid.
        
        Parameters
        ----------
        self : Network object
        edge_length : float
                        length of streets in the network
        
        Returns
        -------
        dictionary
                    keys are nodes, values are (x,y)-coordinate tuples
        """

        pos_dict = {}

        gridsize = self.n_x
        current_node = 0
        for row_idx, row in enumerate(np.arange(0, gridsize, 1)):
            for col_idx, col in enumerate(np.arange(0, gridsize, 1)):
                xval = col * edge_length
                yval = (gridsize - 1 - row) * edge_length
                pos_dict.update({current_node: (xval, yval)})
                current_node += 1

        return pos_dict
