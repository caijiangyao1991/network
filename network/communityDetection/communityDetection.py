# coding=utf8
import datetime
starttime = datetime.datetime.now()
from igraph import Graph as IGraph
import os
import pandas as pd
import numpy as np
from igraph import *

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


#GN算法
# communities = g.community_edge_betweenness(directed=True, weights="weight",clusters=None).as_clustering()
# nodes = [{"name": node["name"]} for node in g.vs]
# for node in nodes:
#     idx = g.vs.find(name=node["name"]).index #找到节点的index
#     node["community"] = communities.membership[idx] #根据index返回cluster群
# result = pd.DataFrame(nodes)
# result.to_csv('./outData/commDetect_GN.csv',encoding='gb18030',index=False)
# modularity = modularity(communities)


#Louvain算法 （Fast Unfolding）
# g = IGraph.TupleList(edges, vertex_name_attr='name',edge_attrs=['weight','relationship'])
# communities = g.community_multilevel(weights="weight")
# print(communities.modularity) #模块度为多少
# print(communities)
# print(max(communities.membership)) #总共有多少类
# nodes = [{"name": node["name"]} for node in g.vs]
# for node in nodes:
#     idx = g.vs.find(name=node["name"]).index #找到节点的index
#     node["community"] = communities.membership[idx] #根据index返回cluster群
# result = pd.DataFrame(nodes)
# result.to_csv('./outData/commDetect_Louvain.csv',encoding='gb18030',index=False)

# #LPA算法
g = IGraph.TupleList(edges, vertex_name_attr='name',edge_attrs=['weight','relationship'])
#假设有一些有标签的数据，没有便签的取成-1
length = len(g.vs)
labels = list(np.array([-1]*len(g.vs)))
fix = np.array([False]*len(g.vs))
for i in range(10,1000):
    labels[i]=0
    fix[i] = True
for i in range(4000,5000):
    labels[i]=1
    fix[i] = True
for i in range(5000,5500):
    labels[i]=2
    fix[i] = True
communities = g.community_label_propagation(initial=labels,fixed=fix)
nodes = [{"name": node["name"]} for node in g.vs]
for node in nodes:
    idx = g.vs.find(name=node["name"]).index #找到节点的index
    node["community"] = communities.membership[idx] #根据index返回cluster群
result = pd.DataFrame(nodes)
result.to_csv('./outData/commDetect_LPA2.csv',encoding='gb18030',index=False)

#infoMap
# g = IGraph.TupleList(edges, vertex_name_attr='name',edge_attrs=['weight','relationship'])
# communities = g.community_infomap(edge_weights='weight')
# print(communities.codelength)
# nodes = [{"name": node["name"]} for node in g.vs]
# for node in nodes:
#     idx = g.vs.find(name=node["name"]).index #找到节点的index
#     node["community"] = communities.membership[idx] #根据index返回cluster群
# result = pd.DataFrame(nodes)
# result.to_csv('./outData/commDetect_infoMap.csv',encoding='gb18030',index=False)


#画图
# plot(communities, mark_groups = True)
# endtime = datetime.datetime.now()
# sec = (endtime - starttime).seconds
# print (sec)