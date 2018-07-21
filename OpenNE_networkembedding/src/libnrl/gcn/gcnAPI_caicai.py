import numpy as np
from .utils import *
from . import models
import time
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd

class GCN(object):

    def __init__(self, graph, learning_rate=0.01, epochs=200,
                 hidden1=16, dropout=0.5, weight_decay=5e-4, early_stopping=10,
                 max_degree=3, clf_ratio=0.1):
        """
                        learning_rate: Initial learning rate
                        epochs: Number of epochs to train
                        hidden1: Number of units in hidden layer 1
                        dropout: Dropout rate (1 - keep probability)
                        weight_decay: Weight for L2 loss on embedding matrix
                        early_stopping: Tolerance for early stopping (# of epochs)
                        max_degree: Maximum Chebyshev polynomial degree
        """
        self.graph = graph
        self.clf_ratio = clf_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden1 = hidden1
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.max_degree = max_degree
        self.preprocess_data()
        self.build_placeholders()
        self.sess = tf.Session()
        # Create model
        self.model = models.GCN(self.placeholders, input_dim=self.features[2][1], hidden1=self.hidden1,
                                weight_decay=self.weight_decay, logging=True)

    def train(self):
        # Init variables 初始化变量
        self.sess.run(tf.global_variables_initializer())
        cost_val = []
        # Train model
        for epoch in range(self.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = self.construct_feed_dict(self.train_mask)
            feed_dict.update({self.placeholders['dropout']: self.dropout})
            #Training step
            outs = self.sess.run([self.model.opt_op, self.model.loss, self.model.accuracy], feed_dict=feed_dict)
            # out = self.sess.run([self.model.outputs], feed_dict=feed_dict1)[0]
            # Validation
            cost, acc, duration = self.evaluate(self.val_mask)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > self.early_stopping and cost_val[-1] > np.mean(cost_val[-(self.early_stopping + 1):-1]):
                print("Early stopping...")
                # saver.save(self.sess, save_path='D:/python/gcn/mode_save/', global_step=epoch)
                self.model.save(self.sess)
                break
        print("Optimization Finished!")
        preds = self.sess.run([self.model.predict()], feed_dict=feed_dict)
        # print(preds[0]['probabilities'])

        # 模型保存
        self.model.save(self.sess)
        # Testing
        test_cost, test_acc, test_duration = self.evaluate(self.test_mask)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    def restore(self):
        self.model.load(self.sess)

    def predict(self,graph1):
        sess = self.sess
        self.preprocess_data_predict(graph1)
        # sess.run(tf.global_variables_initializer())
        feed_dict1 = self.feed_dict_predict()
        preds = sess.run([self.model.predict()], feed_dict=feed_dict1)
        pred_class = preds[0]['classes']
        for i in range(len(pred_class)):
            label_old = self.get_key(self.label_dict,pred_class[i])[0]
            pred_class[i] = label_old
        pred_class_df = pd.DataFrame(pred_class)
        pred_class_df.reset_index(level=0, inplace=True)
        pred_class_df['index'] = pred_class_df['index'].apply(lambda x: int(self.get_key(self.look_back_dict_predict,x)[0]))
        pred_class_df = pred_class_df.sort_values(by=['index'],ascending=[True])
        print(pred_class_df)
        pred_class_df.to_csv('../outdata/classes.csv',header=False,index=False)
        return pred_class_df

    def get_key(self,dict,value):
        return [k for k, v in dict.items() if v == value]

    def preprocess_data(self):
        """adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
            y_train, y_val, y_test can merge to y
                """
        g = self.graph.G
        look_back = self.graph.look_back_list
        self.features = np.vstack([g.nodes[look_back[i]]['feature']
                                   for i in range(g.number_of_nodes())])
        self.features = preprocess_features(self.features)
        self.build_label()
        self.build_train_val_test()
        adj = nx.adjacency_matrix(g)  # the type of graph
        self.support = [preprocess_adj(adj)]

    def preprocess_data_predict(self,graph1):
        g1 = graph1.G
        look_back_predict = graph1.look_back_list
        self.look_back_dict_predict = graph1.look_up_dict
        self.features_predict = np.vstack([g1.nodes[look_back_predict[i]]['feature']
            for i in range(g1.number_of_nodes())])
        self.features_predict = preprocess_features(self.features_predict)
        adj = nx.adjacency_matrix(g1) # the type of graph
        self.support_predict = [preprocess_adj(adj)]


    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = sp.coo_matrix(features)
        features = r_mat_inv.dot(features)
        return sparse_to_tuple(features)

    def build_label(self):
        g = self.graph.G
        look_up = self.graph.look_up_dict
        labels = []
        label_dict = {}
        label_id = 0
        for node in g.nodes():
            labels.append((node, g.nodes[node]['label']))
            for l in g.nodes[node]['label']:
                if l not in label_dict:
                    label_dict[l] = label_id
                    label_id += 1
        self.labels = np.zeros((len(labels), label_id))
        self.label_dict = label_dict
        for node, l in labels:
            node_id = look_up[node]
            for ll in l:
                l_id = label_dict[ll]
                self.labels[node_id][l_id] = 1

    def build_train_val_test(self):
        """
            build train_mask test_mask val_mask
        """
        train_precent = self.clf_ratio
        training_size = int(train_precent * self.graph.G.number_of_nodes())
        state = np.random.get_state()
        np.random.seed(0)
        shuffle_indices = np.random.permutation(np.arange(self.graph.G.number_of_nodes()))
        np.random.set_state(state)

        look_up = self.graph.look_up_dict
        g = self.graph.G
        def sample_mask(begin, end):
            mask = np.zeros(g.number_of_nodes())
            for i in range(begin, end):
                mask[shuffle_indices[i]] = 1
            return mask
        self.train_mask = sample_mask(0, training_size-100)
        self.val_mask = sample_mask(training_size-100, training_size)
        self.test_mask = sample_mask(training_size, g.number_of_nodes())


    def construct_feed_dict(self, labels_mask):
        """Construct feed dictionary."""
        feed_dict = dict()
        feed_dict.update({self.placeholders['labels']: self.labels})
        feed_dict.update({self.placeholders['labels_mask']: labels_mask})
        feed_dict.update({self.placeholders['features']: self.features})
        feed_dict.update({self.placeholders['support'][i]: self.support[i] for i in range(len(self.support))})
        feed_dict.update({self.placeholders['num_features_nonzero']: self.features[1].shape})
        return feed_dict

    def build_placeholders(self):
        """创建占位符"""
        num_supports = 1
        self.placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, self.labels.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }

    def feed_dict_predict(self):
        feed_dict_predict = dict()
        feed_dict_predict.update({self.placeholders['dropout']: self.dropout})
        feed_dict_predict.update({self.placeholders['features']: self.features_predict})
        feed_dict_predict.update({self.placeholders['support'][i]: self.support_predict[i] for i in range(len(self.support_predict))})
        feed_dict_predict.update({self.placeholders['num_features_nonzero']: self.features_predict[1].shape})
        return feed_dict_predict

    # Define model evaluation function
    def evaluate(self, mask):
        t_test = time.time()
        feed_dict_val = self.construct_feed_dict(mask)
        outs_val = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)