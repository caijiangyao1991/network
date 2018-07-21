import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from libnrl.graph import *
from libnrl.classify import Classifier, read_node_label
from libnrl.gcn import gcnAPI_caicai, gcnAPI
import time
import numpy as np
np.set_printoptions(threshold=np.inf)

g = Graph()
g.read_edgelist(filename='../data/cora/cora_edgelist.txt')
g.read_node_label('../data/cora/cora_labels.txt')
g.read_node_features('../data/cora/cora.features')
model = gcnAPI_caicai.GCN(graph=g,epochs=300, clf_ratio=0.5)
model.train()
model.restore()
#预测新数据
g1 = Graph()
g1.read_edgelist(filename='../data/cora/cora_edgelist.txt')
g1.read_node_features('../data/cora/cora.features')
pred_class = model.predict(g1)
# print(pred_class)
# print(pred['probabilities'])

# f = open('../outdata/classes.txt','w')
# for i in pred['classes']:
#     f.write(str(i))
#     f.write('\n')
# f.close()

