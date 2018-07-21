"""安装GEM的包
https://github.com/palash1992/GEM
"""
#encoding='utf-8'
from gem.utils import graph_util
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time
from gem.embedding.hope import HOPE
edge_f = './data/load_rename.csv'
isDirected = True
#load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G.to_directed()
#模型运行
print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
t1 = time()
#根据grid_search调参
embedding = HOPE(d=80, beta=0.01)
# Learn embedding - accepts a networkx graph or file with edge list
Y, t = embedding.learn_embedding(graph=G, is_weighted=True)
print(Y)
# print(Y)
# # Evaluate on graph reconstruction
# MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)


