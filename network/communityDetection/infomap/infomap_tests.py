from two_level_infomap import Infomap
from random_walk import RandomWalk

class Infomap_tests():
    __output_str = ""

    def run_tests(self):
        self.test_calculate_map_equation()
        self.test_random_walk()
        self.test_clusters_reassignment()
        self.test_local_and_cluster_join()
        self.test2_calculate_map_equation()
        return self.__output_str

    def get_graph_for_tests(self):
        graph1 = {'A':['B','C'],
            'B':['A','C'],
            'C':['A','B','D'],
            'D':['C','E','F'],
            'E':['D','F'],
            'F':['E','D']}
        graph2 = {0:[1,3],
                1:[0,2],
                2:[1,3,13],
                3:[0,2,4],
                4:[3,5,7],
                5:[4,6,8],
                6:[5,7],
                7:[4,6],
                8:[5,9,11],
                9:[8,10,12],
                10:[9,11],
                11:[8,10],
                12:[9,13,15],
                13:[2,12,14],
                14:[13,15],
                15:[12,14]}
        return (graph1, graph2)

    def get_visits_probabilities(self):
        visits_probabilities1 = dict.fromkeys(self.get_graph_for_tests()[0],{})
        for node in visits_probabilities1:
            visits_probabilities1[node] = dict.fromkeys(self.get_graph_for_tests()[0],0.071)

        visits_probabilities2 = dict.fromkeys(self.get_graph_for_tests()[1],{})
        for node in visits_probabilities2:
            visits_probabilities2[node] = dict.fromkeys(self.get_graph_for_tests()[1],0.025)

        return (visits_probabilities1, visits_probabilities2)

    def get_partitions(self):
        partition1 = {'A':0,'B':0,'C':0,'D':1,'E':1,'F':1}
        partition2 = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}
        return (partition1,partition2)

    def get_incorrect_partitions(self):
        partition1 = {'A':4,'B':1,'C':1,'D':3,'E':3,'F':3}
        partition2 = {'A':0,'B':2,'C':3,'D':3,'E':5,'F':5}
        return (partition1,partition2)

    def get_map_equations(self):
        map_eq1 = 2.32
        map_eq2 = 4.557
        return (map_eq1, map_eq2)

    def get_partitions_for_cluster_join(self):
        partition1 = {'A':0,'B':0,'C':1,'D':2,'E':2,'F':2}
        partition2 = {'A':0,'B':2,'C':2,'D':1,'E':1,'F':1}
        return (partition1,partition2)

    def get_partitions_for_map_equation(self):
        partition1 = {0: 0, 13: 1, 11: 2, 12: 3, 10: 4, 15: 5, 8: 6, 4: 7,
                        1: 8, 14: 9, 5: 10, 9: 11, 2: 12, 6: 13, 7: 13, 3: 14}
        partition2 = {0: 0, 13: 1, 11: 2, 12: 3, 10: 4, 15: 5, 8: 6, 4: 7,
                        1: 8, 14: 9, 5: 10, 9: 11, 2: 1, 6: 12, 7: 12, 3: 13}
        partition3 = {0: 0, 13: 1, 11: 2, 12: 3, 10: 4, 15: 5, 8: 6, 4: 7,
                        1: 8, 14: 9, 5: 10, 9: 11, 2: 13, 6: 12, 7: 12, 3: 13}
        return (partition1, partition2, partition3)

    def test_calculate_map_equation(self):
        graph = self.get_graph_for_tests()[0]
        rw = RandomWalk(graph)
        rw.walks()
        infomap = Infomap(graph)
        infomap.set_visits_probabilities(rw.get_visits_probabilities())
        for partition, map_eq in zip(self.get_partitions(),self.get_map_equations()):
            if abs(infomap.calculate_map_equation(partition) - map_eq) > 0.2:
                self.__output_str += "Test Infomap1: функция Infomap.calculate_map_equation(clusters_partition) работает некорректно\n"

    def test2_calculate_map_equation(self):
        graph = self.get_graph_for_tests()[1]
        rw = RandomWalk(graph)
        rw.walks()
        infomap = Infomap(graph)
        infomap.set_visits_probabilities(rw.get_visits_probabilities())
        map_equations = []
        for partition in self.get_partitions_for_map_equation():
            map_equations.append(infomap.calculate_map_equation(partition))
        if not all(map_equations[0] > map_equations[i+1] for i in range(len(map_equations) - 1)):
            self.__output_str += "Test Infomap2: функция Infomap.calculate_map_equation(clusters_partition) работает некорректно\n"

    def test_random_walk(self):
        visits_probabilities = ()
        for graph, probabilities in zip(self.get_graph_for_tests(), self.get_visits_probabilities()):
            rw = RandomWalk(graph)
            rw.walks()
            visits_probabilities = rw.get_visits_probabilities()
            for node in visits_probabilities:
                for neighbour in visits_probabilities[node]:
                    if abs(visits_probabilities[node][neighbour] - probabilities[node][neighbour]) > 0.01:
                        self.__output_str += "Test RandomWalk: функция RandomWalk.walks(graph) работает некорректно\n"

    def test_clusters_reassignment(self):
        infomap = Infomap(self.get_graph_for_tests()[0])
        for incorrect_partition in self.get_incorrect_partitions():
            infomap.clusters_reassignment(incorrect_partition)
            for i in range(len(set(incorrect_partition.values()))):
                if i not in incorrect_partition.values():
                    self.__output_str += "Test Infomap: функция Infomap.clusters_reassignment(clusters_partition) работает некорректно\n"

    def test_local_and_cluster_join(self):
        graph = self.get_graph_for_tests()[0]
        rw = RandomWalk(graph)
        rw.walks()
        infomap = Infomap(graph)
        infomap.set_visits_probabilities(rw.get_visits_probabilities())
        cur_partition = self.get_partitions()[1]
        node = 'A'
        cur_map_eq = self.get_map_equations()[1]
        new_map_eq = infomap.local_join(node, cur_map_eq, cur_partition)
        correct_new_map_eq = 4.077
        if cur_partition['A'] != cur_partition['B'] or abs(new_map_eq - correct_new_map_eq) > 0.01:
            self.__output_str += "Test Infomap: функция Infomap.local_join(node, current_map_equation, \
            current_partition) работает некорректно\n"


        for partition in self.get_partitions_for_cluster_join():
            cluster = partition['A']
            old_map_eq = infomap.calculate_map_equation(partition)
            new_map_eq = infomap.cluster_join(cluster, old_map_eq, partition)
            if new_map_eq > old_map_eq:
                self.__output_str += "Test Infomap: функция Infomap.cluster_join(cluster, current_map_equation, \
                current_partition) работает некорректно\n"
                

if __name__=="__main__":
    graph1 = {'A': ['B', 'C'],
              'B': ['A', 'C'],
              'C': ['A', 'B', 'D'],
              'D': ['C', 'E', 'F'],
              'E': ['D', 'F'],
              'F': ['E', 'D']}

    visits_probabilities1 = dict.fromkeys(graph1, {})
    for node in visits_probabilities1:
        visits_probabilities1[node] = dict.fromkeys(graph1, 0.071)

    visits_probabilities2 = dict.fromkeys(graph1, {})
    for node in visits_probabilities2:
        visits_probabilities2[node] = dict.fromkeys(graph1, 0.025)

    from math import log2
    rw = RandomWalk(graph1)
    rw.walks()
    a = rw.get_visits_probabilities()
    print(a)
    infomap = Infomap(graph1)
    infomap.set_visits_probabilities(a)
    partition1 = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 2}

    infomap.calculate_map_equation(partition1)