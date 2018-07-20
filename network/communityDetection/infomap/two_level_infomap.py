from math import log2
from random import sample
import random_walk

class Infomap:
    def __init__(self, graph):
        self.__graph = graph
        self.__number_clusters = 0
        self.__codelength = 0
        self.__visits_probabilities = {}
        self.__nodes_visits = dict.fromkeys(graph.keys(), 0)
        self.__clusters_partition = dict.fromkeys(graph.keys(), 0)

    def clusters_reassignment(self, clusters_partition):
        new_clusters = dict()
        count = 0
        for node in clusters_partition.keys():
            cluster = clusters_partition[node]
            new_cluster = new_clusters.get(cluster)
            if  new_cluster is None:
                new_clusters[cluster] = count
                new_cluster = count
                count += 1
            clusters_partition[node] = new_cluster

    def calculate_map_equation(self, clusters_partition):
        print(clusters_partition)
        number_clusters = len(set(clusters_partition.values()))

        if number_clusters == 1:
            clusters_p_around = 0
            nodes_visits_entropy = 0
            for node in clusters_partition:
                for neighbour in self.__visits_probabilities[node]:
                    clusters_p_around += self.__visits_probabilities[node][neighbour]

            for node in clusters_partition:
                nodes_visits_entropy -= self.__nodes_visits[node] * log2(self.__nodes_visits[node]/clusters_p_around)
            return nodes_visits_entropy

        q_in, index_codebook_entropy, conditional_clusters_entropy = (0, 0, 0)
        clusters_q_in, clusters_p_around, clusters_q_out = [[0 for i in range(number_clusters)] for j in range(3)]
        nodes_visits_entropy = [0 for i in range(number_clusters)]
        for node in clusters_partition:
            for neighbour in self.__visits_probabilities[node]:
                clusters_p_around[clusters_partition[neighbour]] += self.__visits_probabilities[node][neighbour]
                print(clusters_p_around)
                print(clusters_partition)
                if clusters_partition[node] != clusters_partition[neighbour]:
                    #当node和邻居节点不属于一个cluster时
                    clusters_p_around[clusters_partition[node]] += self.__visits_probabilities[node][neighbour]
                    clusters_q_out[clusters_partition[node]] += self.__visits_probabilities[node][neighbour]
                    clusters_q_in[clusters_partition[neighbour]] += self.__visits_probabilities[node][neighbour]
                    q_in += self.__visits_probabilities[node][neighbour]
                print(clusters_p_around)
                print(clusters_q_out)

        for node in clusters_partition:
            nodes_visits_entropy[clusters_partition[node]] += self.__nodes_visits[node] * log2(self.__nodes_visits[node]/clusters_p_around[clusters_partition[node]])

        for i in range(number_clusters):
            index_codebook_entropy -= clusters_q_in[i] * log2(clusters_q_in[i]/q_in)/q_in
            conditional_clusters_entropy -= clusters_q_out[i] * log2(clusters_q_out[i]/clusters_p_around[i]) + nodes_visits_entropy[i]

        return q_in * index_codebook_entropy + conditional_clusters_entropy

    def algorithm(self):
        initial_clusters_partition = {}
        rw = random_walk.RandomWalk(self.__graph)
        rw.walks()
        self.__visits_probabilities = rw.get_visits_probabilities()
        self.__nodes_visits = rw.get_nodes_visits()
        initial_clusters_partition = dict.fromkeys(self.__graph, 0)
        print(self.__nodes_visits)
        for cluster, node in zip(range(len(self.__graph)), self.__graph):
            initial_clusters_partition[node] = cluster
        initial_map_equation = self.calculate_map_equation(initial_clusters_partition)
        best_partition, best_map_equation = (initial_clusters_partition, initial_map_equation)

        was_update = True
        while was_update:
            was_update = False
            cur_partition, cur_map_equation = self.core_algorithm(initial_clusters_partition, initial_map_equation)
            cur_partition, cur_map_equation = self.core_algorithm(cur_partition, cur_map_equation)
            if cur_map_equation < best_map_equation:
                was_update = True
                best_partition, best_map_equation = (cur_partition, cur_map_equation)

        self.__clusters_partition = best_partition
        self.__codelength = best_map_equation
        self.__number_clusters = len(set(best_partition.values()))

    def core_algorithm(self, initial_clusters_partition, initial_map_equation):
        best_clusters_partition = dict(initial_clusters_partition)
        current_clusters_partition = dict(initial_clusters_partition)
        current_map_equation = initial_map_equation
        best_map_equation = initial_map_equation

        was_update = True
        while (was_update):
            was_update = False
            for node in sample(self.__graph.keys(),len(self.__graph)):
                current_map_equation = self.local_join(node, current_map_equation, current_clusters_partition)

            if current_map_equation < best_map_equation:
                best_map_equation = current_map_equation
                best_clusters_partition = dict(current_clusters_partition)
                was_update = True

            current_clusters_partition = dict(initial_clusters_partition)
            current_map_equation = self.calculate_map_equation(current_clusters_partition)

        current_clusters_partition = dict(best_clusters_partition)
        current_map_equation = best_map_equation

        #Вторая фаза
        initial_clusters_partition = dict(current_clusters_partition)
        initial_map_equation = best_map_equation
        was_update = True
        while (was_update):
            was_update = False
            for cluster in sample(set(best_clusters_partition.values()),len(set(best_clusters_partition.values()))):
                current_map_equation = self.cluster_join(cluster, current_map_equation, current_clusters_partition)

            if current_map_equation < best_map_equation:
                best_map_equation = current_map_equation
                best_clusters_partition = dict(current_clusters_partition)
                was_update = True

            current_clusters_partition = dict(initial_clusters_partition)
            current_map_equation = initial_map_equation

        return (best_clusters_partition, best_map_equation)

    def cluster_join(self, cluster, cur_map_eq, cur_partition):
        cur_cluster = None
        cluster_nodes = []
        cluster_neighbours = []
        for node in cur_partition:
            if cur_partition[node] == cluster:
                cluster_nodes.append(node)
        neighbors_clusters = []
        for node in cluster_nodes:
            for neighbour in self.__graph:
                if cluster != cur_partition[neighbour]:
                    neighbors_clusters.append(cur_partition[neighbour])
        neighbors_clusters = tuple(set(neighbors_clusters))

        for neighbor_cluster in neighbors_clusters:
            temp_partition = dict(cur_partition)
            temp_partition[node] = temp_partition[neighbour]

            for node in cluster_nodes:
                temp_partition[node] = neighbor_cluster
            self.clusters_reassignment(temp_partition)
            temp_map_equation = self.calculate_map_equation(temp_partition)
            if temp_map_equation < cur_map_eq:
                cur_cluster = neighbor_cluster
                cur_map_eq = temp_map_equation

        if cur_cluster is not None:
            for node in cluster_nodes:
                cur_partition[node] = cur_cluster
            self.clusters_reassignment(cur_partition)
        return cur_map_eq

    def local_join(self, node, cur_map_eq, cur_partition):
        cur_neighbour = None
        for neighbour in self.__graph[node]:
            if cur_partition[node] != cur_partition[neighbour]:
                temp_partition = dict(cur_partition)
                temp_cluster = temp_partition[node]
                temp_partition[node] = temp_partition[neighbour]
                self.clusters_reassignment(temp_partition)
                temp_map_equation = self.calculate_map_equation(temp_partition)
                if temp_map_equation < cur_map_eq:
                    cur_neighbour = neighbour
                    cur_map_eq = temp_map_equation
        if cur_neighbour is not None:
            temp_cluster = cur_partition[node]
            cur_partition[node] = cur_partition[cur_neighbour]
            self.clusters_reassignment(cur_partition)
        return cur_map_eq

    def get_clusters_partition(self):
        return self.__clusters_partitionget_map_equations

    def get_number_clusters(self):
        return self.__number_clusters

    def get_codelength(self):
        return self.__codelength

    def set_visits_probabilities(self, visits_probabilities):
        self.__visits_probabilities = visits_probabilities
