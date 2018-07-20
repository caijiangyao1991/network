#encoding = utf-8
from igraph import *
import pandas as pd
import numpy as np
import argparse
from math import log
import networkx as nx
import sys

TAU = 0.15
PAGE_RANK = 'page_rank'
MODULE_ID = 'module_id'

def load_and_process_graph(filename):
    """Load the graph, normalize edge weights, compute pagerank, and store all
    this back in node data."""
    # Load the graph
    graph = nx.DiGraph(nx.read_pajek(filename))
    print("loaded a graph (%d nodes, %d edges)" %(len(graph),len(graph.edges())))
    # 计算标准化的权重
    for node in graph:
        edges = graph.edges(node, data=True)
        total_weight = sum([data["weight"] for (_,_,data) in edges])
        for (_,_,data) in edges:
            data["weight"] = data["weight"]/total_weight
    # Get its PageRank, alpha is 1-tau where [RAB2009 says \tau=0.15]
    page_ranks = nx.pagerank(graph, alpha=1 - TAU)
    for (node, page_ranks) in page_ranks.items():
        graph.node[node][PAGE_RANK] = page_ranks  #将pagerank值放到节点的属性中
    return graph

def log2(prob):
    return log(prob,2)

def entropy1(prob):
    if prob==0:
        return 0
    return prob * log2(prob)

class Module:
    def __init__(self, module_id, nodes, graph):
        self.module_id = module_id
        self.nodes = frozenset(nodes)
        self.graph = graph
        self.prop_nodes = 1 - float(len(self.nodes)) / len(graph)
        # print(self.prop_nodes)
        #Set the module_id for every node
        for node in nodes:
            graph.node[node][MODULE_ID] = module_id
        #Compute the total PageRank
        self.total_pr = sum([graph.node[node][PAGE_RANK] for node in nodes])
        #Compute q_out, the exit probability of this module 跳到邻居节点的概率
        self.q_out = self.total_pr * TAU * self.prop_nodes
        for node in self.nodes:
            edges = graph.edges(node, data=True)
            page_rank = graph.node[node][PAGE_RANK]
            if len(edges) == 0:
                self.q_out += page_rank * self.prop_nodes * (1 - TAU)
                continue
            #下一步随机游走者在其邻居节点中
            for (_,dest, data) in edges:
                if dest not in self.nodes:
                    self.q_out += page_rank * data['weight'] * (1 - TAU)
        self.q_plus_p = self.q_out + self.total_pr

    def get_codebook_length(self):
        "Computes module codebook length according to [RAB2009, eq. 3]"
        first = -entropy1(self.q_out / self.q_plus_p)
        second = -sum( \
            [entropy1(self.graph.node[node][PAGE_RANK] / self.q_plus_p) \
             for node in self.nodes])
        return (self.q_plus_p) * (first + second)


class Clustering:
    "Stores a clustering of the graph into modules"
    def __init__(self, graph, modules):
        self.graph = graph
        self.total_pr_entropy = sum([entropy1(graph.node[node][PAGE_RANK]) for node in graph])
        self.modules = [Module(module_id, module, graph) \
                for (module_id, module) in enumerate(modules)]

    def get_mdl(self):
        "Compute the MDL of this clustering according to [RAB2009, eq. 4]"
        total_qout = 0
        total_qout_entropy = 0
        total_both_entropy = 0
        for mod in self.modules:
            q_out = mod.q_out
            total_qout += q_out
            total_qout_entropy += entropy1(q_out)
            total_both_entropy += entropy1(mod.q_plus_p)
        term1 = entropy1(total_qout)
        term2 = -2 * total_qout_entropy
        term3 = -self.total_pr_entropy
        term4 = total_both_entropy
        return term1 + term2 + term3 + term4

    def get_index_codelength(self):
        "Compute the index codebook length according to [RAB2009, eq. 2]"
        if len(self.modules) == 1:
            return 0
        total_q = sum([mod.q_out for mod in self.modules])
        entropy = -sum([entropy1(mod.q_out / total_q) for mod in self.modules])
        return total_q * entropy

    def get_module_codelength(self):
        "Compute the module codebook length according to [RAB2009, eq. 3]"
        return sum([mod.get_codebook_length() for mod in self.modules])

def print_tree_file(graph, modules):
    """Produces a .tree file from the given clustering that is compatible with
    the InfoMapCheck utility."""
    for (mod_id, mod) in enumerate(modules):
        for (node_id, node) in enumerate(mod):
            print( "%d:%d %f \"%s\"" % (mod_id+1, node_id+1,
                    graph.node[node][PAGE_RANK], node))

def main():
    "Read the supplied graph and modules and output MDL"
    # Read the arguments
    # parser = argparse.ArgumentParser(description="Calculate the infomap")
    # parser.add_argument('-g', '--graph-filename', type=argparse.FileType('r'),
    #         help="the .net file to use as the graph", required=True)
    # parser.add_argument('-m', '--module-filename', default="2009_figure3a.mod",
    #         help="the .mod file to use as the clustering")
    # options = parser.parse_args(argv[1:])

    #TODO 读入图
    graph = load_and_process_graph('./data/2009_figure3ab.net')

    #TODO single_nodes is the "trivial" module mapping 每个节点自己作为一个社群
    single_nodes = [[nodes] for nodes in graph]
    #Using default clustering of every node in its own module
    modules = single_nodes
    clustering = Clustering(graph, modules)
    print("This clustering has MDL %.2f (Index %.2f, Module %.2f)" % \
    (clustering.get_mdl(), clustering.get_index_codelength(),
     clustering.get_module_codelength()))


if __name__ == "__main__":
    main()





