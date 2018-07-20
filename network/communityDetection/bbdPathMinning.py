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
#TODO stage1:先构造无向图
g = IGraph.TupleList(edges, vertex_name_attr='name',edge_attrs=['weight','relationship'])
# for u in g.es:
#     s,v = u.tuple
#     print(g.es.select(s,v)["relationship"])
#TODO stage2:遍历所有节点的所有路径形成特性route列表和每个节点的所有最短路径字典nodePath
# 某个节点的所有路径
relationship = g.es["relationship"]
route = []
nodeRoute = []
routeid = []
nodePath = {}
for nodes in g.vs:
    paths = g.get_all_shortest_paths(nodes)
    for p in paths:
        plist = g.get_eids(path=p)
        plist2 = [relationship[x] for x in plist]
        if len(plist2)>0 and plist2 not in route:
            route.append(plist2)
        if len(plist2) > 0 and plist2 not in nodeRoute:
            nodeRoute.append(plist2)
    nodePath[nodes['name']] = nodeRoute
    nodeRoute = []
print("nodePath:" + str(nodePath))

#TODO stage3:构造训练集矩阵
num_samples = len(g.vs)
print("num_samples:"+ str(num_samples))
pathLen = len(route)
print("pathLen:"+ str(pathLen))
#nodeDict
nodeName =[]
nodeIndex =[]
for v in g.vs:
    nodeName.append(v['name'])
    nodeIndex.append(v.index)
nodeDict = dict(zip(nodeIndex, nodeName))
print(nodeName)
#pathDict
pathDict = {}
for index, value in enumerate(route):
    pathDict[index] = value
print("pathDict:"+str(pathDict))

matrix = np.zeros((num_samples, pathLen), np.float32)
for i in range(num_samples):
    for j in range(pathLen):
        nodes = nodeDict[i]
        #得到该节点的路径
        paths = nodePath[nodes]
        if pathDict[j] in paths:
            matrix[i][j] = 1.0
print(matrix)
# #
# #
