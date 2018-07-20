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


# #中心度计算、紧密中心性、介数中心性、pageRank等计算
# print ('STARTING TIME : ', time.asctime((time.localtime(time.time()))))
# data = []
# for p in zip(g.vs, g.degree(), g.indegree(), g.outdegree(),g.closeness(weights="weight"),g.betweenness(),g.pagerank(weights="weight",niter=1000)):
#     data.append({'name':p[0]['name'], 'degree':p[1], 'indegree':p[2], 'outdegree':p[3], 'closeness':p[4]
#                    ,'betweeness':p[5],'pageRank':p[6]})
#
# #根据pagerank排个序
# result = sorted(data, key=lambda x:x['pageRank'],reverse=True)
# result1 = pd.DataFrame.from_dict(result)
# result1.to_csv('./outData/graphIndex.csv',index=False,encoding='gb18030')
# print('ENDING TIME: ', time.asctime((time.localtime(time.time()))))
#
# #某个节点的所有路径
# paths = g.get_all_shortest_paths(u"汕头市天昊服饰有限公司",weights="weight")
# names = g.vs["name"]
# relationship = g.es["relationship"]
# for p in paths:
#     # 节点的id
#     print(p)
#     name = [names[x] for x in p]
#     allpath = []
#     print([names[x] for x in p])
#     plist = g.get_eids(path=p) #可以得到经过某条路径的id
#     print(plist)
#     print([relationship[x] for x in plist])
#     for x,y in zip(p, plist):
#         allpath.append(names[x])
#         allpath.append(relationship[y])
#     print(allpath)



# #手工计算介数中心性
# sp = []
# target = 22
# for v in g.vs:
#     paths = g.get_all_shortest_paths(v["name"])
#     for p in paths:
#         if target in p and target!= p[-1]:
#             sp.append(p)
# #去重i到j和j到i的同一条路径
# spbt = 0
# tu = []
# for x in sp:
#     if set((x[0],x[-1])) not in tu:
#         tu.append(set((x[0],x[1])))
#         spbt += 1
# print(spbt)
#
# #边介数计算
# btes = []
# names = g.vs["name"]
# for p in zip(g.es(), g.edge_betweenness()):
#     e = p[0].tuple # .tuple可以得到这条边的两个节点，.index可以得到这条边的id
#     print(e.tuple)
#     btes.append({"edge":(names[e[0]],names[e[1]]),"bt":p[1]})

#完全子图
# clique = []
# clis = set()
# names = g.vs["name"]
# cliques = g.cliques(min=3)
# # cliques = g.largest_cliques()
# print(cliques)
# for cli in cliques:
#     for x in cli:
#         clis.add(names[x])
#     clique.append(clis)
#     clis = set()
# print(clique)




















