from os.path import join
import numpy as np
import networkx as nx
from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class GithubStargazersGenerator(Generator):

    def init(self):
        self.data_path = self.local_config['parameters']['data_path']
        self.max_node_count = self.local_config['parameters']['max_node_count']
        self.num_count_divisibility = self.local_config['parameters']['node_count_divisibility']
        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # set defaults
        local_config['parameters']['data_path'] = local_config['parameters'].get('data_path','data/datasets/github_stargazers')
        local_config['parameters']['max_node_count'] = local_config['parameters'].get('max_node_count', -1)
        local_config['parameters']['node_count_divisibility'] = local_config['parameters'].get('node_count_divisibility', -1)

    def generate_dataset(self):
        if len(self.dataset.instances):
            return

        adj_matrix_path = join(self.data_path, 'github_stargazers_A.txt')
        graph_indicator_path = join(self.data_path, 'github_stargazers_graph_indicator.txt')
        graph_labels_path = join(self.data_path, 'github_stargazers_graph_labels.txt')

        edges = np.loadtxt(adj_matrix_path, delimiter=',',dtype=int)
        graph_indicator = np.loadtxt(graph_indicator_path, dtype=int)
        graph_labels = np.loadtxt(graph_labels_path, dtype=int)

        graph_ids, node_counts = np.unique(graph_indicator, return_counts=True)

        # Filtered has the indices of graph_ids for which a graph will be created
        filtered = np.arange(len(node_counts)) # Thake them all to start
        if self.max_node_count > -1:
            filtered_relative = np.where(node_counts[filtered] < self.max_node_count)[0]
            filtered = filtered[filtered_relative]
        if self.node_count_divisibility > -1:
            filtered = np.where(node_counts[filtered] % self.node_count_divisibility == 0)[0]
            filtered = filtered[filtered_relative]

        self.context.logger.info(f"Generating {len(filtered)} graphs")
        for iteration, graph_id in enumerate(graph_ids[filtered], start=1):
            if iteration % 200 == 0:
                self.context.logger.info(f"Generating graph {iteration} with id {graph_id}")

            graph_nodes = np.where(graph_indicator == graph_id)[0] + 1 # Add one to make up for the 0th index
            graph_edges = edges[np.isin(edges, graph_nodes).any(axis=1)]

            G = nx.Graph()
            G.add_edges_from(graph_edges)

            label = graph_labels[graph_id - 1]
            self.dataset.instances.append(GraphInstance(id=graph_id, data=nx.to_numpy_array(G), label=label))

    def get_num_instances(self):
        return len(self.dataset.instances)