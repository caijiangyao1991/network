import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from libnrl.graph import *
from libnrl import node2vec
from libnrl.classify import Classifier, read_node_label
import matplotlib as plt

g = Graph()
g.read_edgelist(filename='../data/load_rename.csv', weighted=True, directed=True)
#调参
X, Y = read_node_label('../data/load_label.csv')
tuned_parameters = {'path_length':[20,100],'num_paths':[10,20,50],'dim':[30,80,200],'p':[0.25, 0.5, 1, 2, 4],'q':[0.25, 0.5, 1, 2, 4]}
test_scores = {}
for p in tuned_parameters['p']:
    for q in tuned_parameters['q']:
        model = node2vec.Node2vec(graph=g, path_length=80, num_paths=10, dim=30,p=p, q=q, window=20)
        vectors = model.vectors
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        score = clf.split_train_evaluate(X, Y, 0.7)
        test_scores[(p,q)] = score['micro']
print(test_scores)
scorebest = max(test_scores.keys(),key=lambda s:test_scores[s])
print(scorebest)
#选取参数

model.save_embeddings('../outdata/load_embed.txt')


