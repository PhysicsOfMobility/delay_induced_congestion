import networkx as nx
import numpy as np


class Network:
    """Streets are edges of a directed networkx graph.
    """
    
    def __init__(self, env, num_nodes = 25, t_0 = 1, N_0 = 10, periodic = True):
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
                                       t_0 = t_0, 
                                       N_0 = N_0,
                                       numcars = 0,
                                       traveltime = t_0)
            dir_periodic_grid.add_edge(edge[1], edge[0], 
                                       t_0 = t_0, 
                                       N_0 = N_0,
                                       numcars = 0,
                                       traveltime = t_0)
        
        if periodic:
            # Add edges connecting the sides and top and bottom, to ensure periodicity.
            for node in list(dir_periodic_grid.nodes()):
                if dir_periodic_grid.out_degree(node) < 4:
                    if node % gridsize == 0:
                        dir_periodic_grid.add_edge(node, node+gridsize-1, 
                                                   t_0 = t_0, 
                                                   N_0 = N_0,
                                                   numcars = 0,
                                                   traveltime = t_0)
                    if (node+1) % gridsize == 0:
                        dir_periodic_grid.add_edge(node, node-gridsize+1, 
                                                   t_0 = t_0, 
                                                   N_0 = N_0,
                                                   numcars = 0,
                                                   traveltime = t_0)
                    if node-gridsize < 0:
                        dir_periodic_grid.add_edge(node, node + (gridsize-1)*gridsize, 
                                                   t_0 = t_0, 
                                                   N_0 = N_0,
                                                   numcars = 0,
                                                   traveltime = t_0)
                    if node+gridsize >= num_nodes:
                        dir_periodic_grid.add_edge(node, node - (gridsize-1)*gridsize, 
                                                   t_0 = t_0, 
                                                   N_0 = N_0,
                                                   numcars = 0,
                                                   traveltime = t_0)
        
        self.graph = dir_periodic_grid    
        self.edges = list(dir_periodic_grid.edges())
        self.n_x = gridsize
        self.n_y = gridsize
        self.num_nodes = num_nodes
        self.env = env
        
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
        """Total time needed to traverse a path of streets;
        estimated based on averaged street load information.
        """
        time = 0
        listlength = self.env.Tav / self.env.av_resolution
        for edge in path:
            t_0 = self.graph[edge[0]][edge[1]]['t_0']
            N_0 = self.graph[edge[0]][edge[1]]['N_0']
            Nav = self.env.numcars_dict[edge]["num_sum"] / listlength
            if Nav == 0:
                time += t_0
            else:
                time += t_0 * N_0 * (np.exp(Nav / N_0) - 1) / Nav
        return time
