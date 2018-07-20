#encoding = utf8
import numpy as np

# state_needed = set(["mt","wa","or", "id", "nv", "ut", "ca", "az"])
#
# #广告台清单
# stations = {}
# stations["kone"] = set(["id","nv","ut"])
# stations["ktwo"] = set(["wa", "id", "mt"])
# stations["kthree"] = set(["or", "nv", "ca"])
# stations["kfour"] = set(["nv", "ut"])
# stations["kfive"] = set(["ca", "az"])
#
# final_stations = set()
#
# while state_needed:
#     best_station = None
#     state_covered = set()
#     for station, states in stations.items():
#         covered = state_needed & states
#         if len(covered)>len(state_covered):
#             state_covered=covered
#             best_station = station
#     state_needed = state_needed-state_covered
#     final_stations.add(best_station)

def calculate_map_equation(self, clusters_partition):
    number_clusters = len(set(clusters_partition.values()))

    if number_clusters == 1:
        clusters_p_around = 0
        nodes_visits_entropy = 0
        for node in clusters_partition:
            for neighbour in self.__visits_probabilities[node]:
                clusters_p_around += self.__visits_probabilities[node][neighbour]

        for node in clusters_partition:
            nodes_visits_entropy -= self.__nodes_visits[node] * log2(self.__nodes_visits[node] / clusters_p_around)
        return nodes_visits_entropy

    q_in, index_codebook_entropy, conditional_clusters_entropy = (0, 0, 0)
    clusters_q_in, clusters_p_around, clusters_q_out = [[0 for i in range(number_clusters)] for j in range(3)]
    nodes_visits_entropy = [0 for i in range(number_clusters)]
    for node in clusters_partition:
        for neighbour in self.__visits_probabilities[node]:
            clusters_p_around[clusters_partition[neighbour]] += self.__visits_probabilities[node][neighbour]
            if clusters_partition[node] != clusters_partition[neighbour]:
                clusters_p_around[clusters_partition[node]] += self.__visits_probabilities[node][neighbour]
                clusters_q_out[clusters_partition[node]] += self.__visits_probabilities[node][neighbour]
                clusters_q_in[clusters_partition[neighbour]] += self.__visits_probabilities[node][neighbour]
                q_in += self.__visits_probabilities[node][neighbour]

    for node in clusters_partition:
        nodes_visits_entropy[clusters_partition[node]] += self.__nodes_visits[node] * log2(
            self.__nodes_visits[node] / clusters_p_around[clusters_partition[node]])

    for i in range(number_clusters):
        index_codebook_entropy -= clusters_q_in[i] * log2(clusters_q_in[i] / q_in) / q_in
        conditional_clusters_entropy -= clusters_q_out[i] * log2(clusters_q_out[i] / clusters_p_around[i]) + \
                                        nodes_visits_entropy[i]

    return q_in * index_codebook_entropy + conditional_clusters_entropy
