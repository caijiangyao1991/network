# from DiseaseNetwork import DiseaseState
from DiseaseNetwork import DiseaseNode
from DiseaseNetwork import DiseaseNetwork
from random import random
# from threading import Timer
from threading import Thread
from time import sleep


class SIR:
    class __Timer:
        def __init__(self):
            self.__calling_method = None  # type: function
            self.__stop_method = None  # type: function
            self.__stopConditionFunc = None  # type: function
            self.__period = 0  # type: int
            self.__thread = Thread(target=self.__run, args=())  # type : Thread
            self.__stop = False  # type: bool

        def setStartFunc(self, start_func):
            self.__calling_method = start_func

        def setStopFunc(self, stop_func):
            self.__stop_method = stop_func

        def setStopConditionFunc(self, stop_condition_func):
            self.__stopConditionFunc = stop_condition_func

        def setSpan(self, seconds):
            self.__period = seconds

        def start(self):
            if self.__period > 0:
                self.__thread.start()
            else:
                self.__run(threaded=False)

        def stop(self):
            self.__stop = True

        def __run(self, threaded=True):
            while (not threaded) or (not self.__stop):
                self.__calling_method()
                if self.__stopConditionFunc():
                    break
                sleep(self.__period)
            self.__stop_method()

    def __init__(self, beta, mu, time_step=1):
        """ Constructor.
        :param beta: is infect rate.
        :type beta: float
        :param mu: is recover rate.
        :type mu: float
        :param time_step: is a delay between steps. (in seconds)
        :type time_step: int
        """
        self.beta = beta                    # type: float
        self.mu = mu                        # type: float
        self.diseaseNet = DiseaseNetwork()  # type: DiseaseNetwork
        self.timeStep = time_step           # type: int
        self.__currentTime = 0              # type: int
        self.__stop = False                 # type: bool
        self.__infectedNodes = list()       # type: list[int]

    def setDiseaseNetwork(self, d_network):
        self.diseaseNet = d_network

    def go(self):
        self.__currentTime += 1

        __ta = round(1.0 / self.mu)
        __recovered_nodes = list()
        __new_infected_nodes = list()
        for __node_id in self.__infectedNodes:
            __node_obj = self.diseaseNet.getNode(__node_id)
            if (self.__currentTime - __node_obj.infectedTime) >= __ta:
                __node_obj.recover()
                # self.__infectedNodes.remove(__node_id)
                __recovered_nodes.append(__node_id)
            else:
                __susceptible_neighbors = list()
                for __neighbor_id in __node_obj.neighbors:
                    if self.diseaseNet.getNode(__neighbor_id).isSusceptible():
                        __susceptible_neighbors.append(__neighbor_id)
                __new_infected_nodes.extend(self.infect(__susceptible_neighbors))

        for __node_id in __recovered_nodes:
            self.__infectedNodes.remove(__node_id)
        self.__infectedNodes.extend(__new_infected_nodes)

        if self.__currentTime % __ta == 0:
            print ("Iteration =", self.__currentTime)

    def __stopped(self):
        print ("SIR run is terminated...")

    def __isDone(self):
        return len(self.__infectedNodes) == 0

    def infect(self, susceptible_nodes):
        """ Infects the input susceptible nodes with probability of self.beta
        :param susceptible_nodes: is a list of node ids which their state is DiseaseState.Susceptible.
        :type susceptible_nodes: list[int]
        :return: Returns list of disease nodes which they infected in this method.
        :rtype: list[int]
        """
        __infected_nodes = list()   # type: list[int]
        for __node_id in susceptible_nodes:
            __r = random()
            if __r > self.beta:
                continue
            __node_obj = self.diseaseNet.getNode(__node_id)
            __node_obj.infect(self.__currentTime)
            __infected_nodes.append(__node_obj.id)
        return __infected_nodes

    def start(self):
        __timer = SIR.__Timer()
        __timer.setStartFunc(self.go)
        __timer.setStopFunc(self.__stopped)
        __timer.setStopConditionFunc(self.__isDone)
        __timer.setSpan(self.timeStep)
        __timer.start()

    def getTime(self):
        return self.__currentTime * self.timeStep

    def setSeeds(self, seeds):
        self.__infectedNodes.extend(seeds)
        for __node_id in self.__infectedNodes:
            self.diseaseNet.getNode(__node_id).infect(0)

    def get_dik_dt(self, k, t=None, C=0):
        __nodes_k = 0
        __suspt_k = 0
        __infec_k = 0
        __nghbr_n_k = 0
        __nghbr_i_k = 0
        for __node_id, __node_obj in self.diseaseNet.nodes.items():
            if __node_obj.getDegree() == k:
                __nodes_k += 1
                if __node_obj.isSusceptible():
                    __suspt_k += 1
                elif __node_obj.isInfected():
                    __infec_k += 1
                for __nghbr_id in __node_obj.neighbors:
                    __nghbr_n_k += 1
                    if self.diseaseNet.nodes[__nghbr_id].isInfected():
                        __nghbr_i_k += 1
        __infect_part = self.beta * __suspt_k / __nodes_k
        __recovr_part = self.mu * __nghbr_i_k / __nghbr_n_k
        if t is None:
            return __infect_part - __recovr_part
        else:
            return (__infect_part - __recovr_part) * t + C

    def isConverged(self):
        return len(self.__infectedNodes) == 0

    def getInfectedNodes(self):
        return self.__infectedNodes
