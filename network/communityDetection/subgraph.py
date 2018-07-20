#coding=utf8
from igraph import *
from igraph import Graph as IGraph
import os
import time
import pandas as pd
import numpy as np
edges = []
path = './data/'
firstline=True
for file in os.listdir(path):
    if file.endswith(".csv"):
        with open(path+file, 'rb') as f:
            for row in f.readlines():
                if firstline == True:
                    firstline = False
                    continue
                parts = row.decode().replace(' ','').replace('\r\n','').strip().split(",")
                try:
                    u, v, e, weight = [i for i in parts]
                    edges.append((u, v, int(weight), e))
                except ValueError:
                    continue
g = IGraph.TupleList(edges, directed=True, vertex_name_attr='name',edge_attrs=['weight','relationship'])

#连通子图
subgraph = g.components(mode=WEAK)
nodes = [{"name": node["name"]} for node in g.vs]
graph = {}
for node in nodes:
    idx = g.vs.find(name=node["name"]).index
    node["subgraph"] = subgraph.membership[idx]
    if node["subgraph"] not in graph:
        graph[node["subgraph"]] = [node["name"]]
    else:
        graph[node["subgraph"]].append(node["name"])
graphs = {}
for key, value in graph.items():
    if len(value)>3:
        graphs[key] = value
print(len(graphs))
print(graphs)
graph_list = sorted(graphs.items(), key=lambda f: len(f[1]),reverse=False)
print(graph_list)
