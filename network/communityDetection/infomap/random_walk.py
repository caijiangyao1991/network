from random import randint, choice
from math import log2

class RandomWalk:
    def __init__(self, graph, number_walks=0, number_steps=0):
        self.__graph = graph
        if not number_walks:
            self.__number_walks = 10 * int(log2(len(graph)))
        else:
            self.__number_walks = number_walks
        if not number_steps:
            self.__number_steps = 10000 * int(log2(len(graph)))
        else:
            self.__number_steps = number_steps
        self.__visits_probabilities = dict.fromkeys(graph, {})
        for node in graph:
            self.__visits_probabilities[node] = dict.fromkeys(graph[node], 0)
        self.__nodes_visits = dict.fromkeys(graph, 0)

    def walk(self, start=None):
        current = choice(list(self.__graph.keys())) if start is None else start

        for i in range(self.__number_steps):
            successor = choice(self.__graph[current])
            self.__visits_probabilities[current][successor] += 1
            current = successor


    def walks(self):
        for i in range(0, self.__number_walks):
            self.walk()

        for node in self.__visits_probabilities:
            for neighbour in self.__visits_probabilities[node]:
                self.__visits_probabilities[node][neighbour] /= self.__number_walks * self.__number_steps

        for node in self.__visits_probabilities:
            for neighbour in self.__visits_probabilities[node]:
                self.__nodes_visits[neighbour] += self.__visits_probabilities[node][neighbour]

    def get_nodes_visits(self):
        return self.__nodes_visits

    def get_visits_probabilities(self):
        return self.__visits_probabilities
