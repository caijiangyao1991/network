"""安装GEM的包
https://github.com/palash1992/GEM
"""
#encoding='utf-8'
from gem.utils import graph_util
from gem.evaluation import evaluate_node_classification as ev
from time import time
from gem.embedding.sdne import SDNE
from gem.evaluation import visualize_embedding as viz
disp_avlbl = True
import os
# if 'DISPLAY' not in os.environ:
#     disp_avlbl = False
#     import matplotlib
#     matplotlib.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

edge_f = './data/load_rename.csv'
isDirected = True
#load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G.to_directed()
#模型运行
print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
t1 = time()
#根据grid_search调参
embedding = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=500,
                modelfile=['./outdata/intermediate/enc_model.json', './outdata/intermediate/dec_model.json'],
                weightfile=['./outdata/intermediate/enc_weights.hdf5', './outdata/intermediate/dec_weights.hdf5'])

# Learn embedding - accepts a networkx graph or file with edge list
Y, t = embedding.learn_embedding(graph=G, is_weighted=True)

# ev.evaluateNodeClassification()
# print(Y)

viz.plot_embedding2D(embedding.get_embedding(),
                         di_graph=G, node_colors=None)
plt.show()
# print(Y)
# # Evaluate on graph reconstruction
# MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)


