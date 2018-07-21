from random import sample
import copy


class DiseaseState(object):
    Susceptible = 1
    Infected = 2
    Recovered = 3


class DiseaseNode:
    def __init__(self, id):
        self.id = id                            # type: int
        self.infectedTime = -1                  # type: int
        self.state = DiseaseState.Susceptible   # type: int
        self.neighbors = list()                 # type: list[int]

    def infect(self, infected_time):
        self.infectedTime = infected_time
        self.state = DiseaseState.Infected

    def recover(self):
        self.state = DiseaseState.Recovered

    def addNeighbor(self, neighbor_id):
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)

    def isInfected(self):
        return self.state == DiseaseState.Infected

    def isSusceptible(self):
        return self.state == DiseaseState.Susceptible

    def isRecovered(self):
        return self.state == DiseaseState.Recovered

    def getNeighbors(self):
        return copy.deepcopy(self.neighbors)

    def getDegree(self):
        return len(self.neighbors)


class DiseaseNetwork:
    def __init__(self):
        self.nodes = dict()         # type: dict[int, DiseaseNode]

    def addNode(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = DiseaseNode(node_id)
            return self.nodes[node_id]
        return None

    def getNode(self, node_id):
        """ Gets node which has id of node_id.
        :type node_id: int
        :return: Returns None when there is no node with id of node_id, or DiseaseNode object.
        :rtype: None|DiseaseNode
        """
        if node_id in self.nodes:
            return self.nodes[node_id]
        return None

    def size(self):
        return len(self.nodes)

    def getRandomNodes(self, k):
        __ids = list(self.nodes.keys())
        __rand_indexes = sample(range(len(__ids)), k)
        __node_ids = list()
        for __rand_index in __rand_indexes:
            __node_ids.append(__ids[__rand_index])
        return __node_ids

    def getRecoveredNodes(self):
        __recovered_nodes = list()  # type: list[DiseaseNode]
        for __node_id, __node_obj in self.nodes.items():
            if __node_obj.isRecovered():
                __recovered_nodes.append(__node_obj)
        return __recovered_nodes

    def getSusceptibleNodes(self):
        __susceptible_nodes = list()  # type: list[DiseaseNode]
        for __node_id, __node_obj in self.nodes.items():
            if __node_obj.isSusceptible():
                __susceptible_nodes.append(__node_obj)
        return __susceptible_nodes

    def getInfectedNodes(self):
        __infected_nodes = list()  # type: list[DiseaseNode]
        for __node_id, __node_obj in self.nodes.iteritems():
            if __node_obj.isInfected():
                __infected_nodes.append(__node_obj)
        return __infected_nodes

    def getNodes(self, deg, state=None):
        """ Returns nodes of degree deg.
        :param deg: nodes degree.
        :type deg: int
        :return: Returns list of nodes which are of degree deg.
        :rtype: list[int]
        """
        __node_ids = list()
        for __node_id, __node_obj in self.nodes.iteritems():
            if __node_obj.getDegree() == deg:
                if state is None:
                    __node_ids.append(__node_id)
                elif __node_obj.state == state:
                    __node_ids.append(__node_id)
        return __node_ids

    # def get_dik_dt(self, k):
    #     __nodes_k = 0
    #     __suspt_k = 0
    #     __infec_k = 0
    #     __nghbr_n_k = 0
    #     __nghbr_i_k = 0
    #     for __node_id, __node_obj in self.nodes.iteritems():
    #         if __node_obj.getDegree() == k:
    #             __nodes_k += 1
    #             if __node_obj.isSusceptible():
    #                 __suspt_k += 1
    #             elif __node_obj.isInfected():
    #                 __infec_k += 1
    #             for __nghbr_id in __node_obj.neighbors:
    #                 __nghbr_n_k += 1
    #                 if self.nodes[__nghbr_id].isInfected():
    #                     __nghbr_i_k += 1
    #


