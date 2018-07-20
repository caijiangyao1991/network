
# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-06-24 14:56:40
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-06-27 16:14:00

import json

class Edge():
    def __init__(self, src, dest, weight):
        self.src = src
        self.dest = dest
        self.weight = weight

class LabelProp():

    def __init__(self):
        self.initialize_env()

    ################################################################################
    #   Prepare Data
    ################################################################################

    def initialize_env(self):
        self.vertex_adj_map = {}  # int: [Edge]
        self.vertex_in_adj_map = {}  # int: float
        self.vertex_deg_map = {}  # int, float
        self.vertex_label_map = {}  # int, int
        self.label_index_map = {}  # int, int
        self.vertex_f_map = {}  # int, [float]
        self.vertex_size = 0
        self.label_size = 0
        self.labelled_size = 0

    def setup_env(self):

        # initialize vertex_in_adj_map
        for vertex_id in self.vertex_adj_map.keys():
            if vertex_id not in self.vertex_in_adj_map:
                self.vertex_in_adj_map.setdefault(vertex_id, [])

        # setup vertex_in_adj_map
        for vertex_id in self.vertex_in_adj_map.keys():
            for edge in self.vertex_adj_map[vertex_id]:
                self.vertex_in_adj_map[edge.dest].append(edge)
        print("vertex_in_adj_map:" + str(self.vertex_in_adj_map))

        # setup vertex_deg_map
        for vertex_id in self.vertex_adj_map.keys():
            degree = .0
            if vertex_id in self.vertex_deg_map:
                degree = self.vertex_deg_map[vertex_id]
            for edge in self.vertex_adj_map[vertex_id]:
                degree += edge.weight
            self.vertex_deg_map[vertex_id] = degree
        print("vertex_deg_map:" + str(self.vertex_deg_map))

        # setup vertex_f_map
        v_set = self.vertex_label_map.keys()
        l_set = self.vertex_label_map.values()
        l_set = list(set(l_set))
        l_set.sort()

        label_enum = 0
        for l in l_set:
            if int(l) == 0:
                continue
            self.label_index_map[l] = label_enum
            label_enum += 1
        self.label_size = label_enum
        print("label_index_map:"+ str(self.label_index_map))

        self.labelled_size = 0
        for v in v_set:
            arr = []
            l = int(self.vertex_label_map[v])
            if l == 0:
                # unlabelled
                for i in range(label_enum):
                    arr.append(.0)
            else:
                # labelled
                self.labelled_size += 1
                ix = int(self.label_index_map[self.vertex_label_map[v]])
                for i in range(label_enum):
                    if i == ix:
                        arr.append(1.)
                    else:
                        arr.append(0.)
            self.vertex_f_map.setdefault(v, arr)
        print("vertex_f_map:"+str(self.vertex_f_map))

    def load_data_from_file(self, filename):
        with open(filename, 'rb') as f:
            lines = [eval(_.strip()) for _ in f.readlines()]
            self.load_data_from_mem(lines)

    def load_data_from_mem(self, data):
        self.initialize_env()
        for line in data:
            self.process_data_line(line)
        self.setup_env()

    def process_data_line(self, line):
        # [vertexId, vertexLabel, [edges]]
        # unlabeled vertex if vertexLabel == 0
        # i.e. [2, 1, [[1, 1.0], [3, 1.0]]]
        try:
            vertex_id = line[0]
            vertex_label = line[1]
            edges = line[2]
            edge_list = []
            self.vertex_label_map.setdefault(vertex_id, vertex_label)
            print("vertex_label_map:" + str(self.vertex_label_map))
            for edge in edges:
                dest_vertex_id = int(edge[0])
                edge_weight = float(edge[1])
                edge_list.append(Edge(vertex_id, dest_vertex_id, edge_weight))
            self.vertex_adj_map.setdefault(vertex_id, edge_list)
            print("vertex_adj_map:" + str(self.vertex_adj_map))
        except Exception as e:
            raise Exception("Coundn't parse vertex from line")

    ################################################################################
    #   Label Propagation
    ################################################################################

    def debug(self):
        labels = []
        for label in self.label_index_map.keys():
            labels.insert(int(self.label_index_map[label]), label)
        ans = []
        print("vertex_f_map："+ str(self.vertex_f_map))
        for vertex_id in self.vertex_f_map.keys():
            arr = self.vertex_f_map[vertex_id]
            max_f_val = .0
            max_f_val_idx = 0

            im_ans = [vertex_id]
            for i in range(len(labels)):
                f_val = arr[i]
                if f_val > max_f_val:
                    max_f_val = f_val
                    max_f_val_idx = i
                im_ans.append([labels[i], arr[i]])
            print(labels[max_f_val_idx])
            im_ans.insert(1, labels[max_f_val_idx])
            print("im_ans:" + str(im_ans))
            ans.append(im_ans)
        return ans

    def iterate(self):
        next_vertex_f_map = {}  # int, [double]
        diff = 0

        for vertex_id in self.vertex_f_map.keys():
            if self.vertex_label_map[vertex_id]:  # skip labelled
                continue

            # update F(vertex_id) .. vertex_f_map
            next_f_value = []  # double
            f_values = self.vertex_f_map[vertex_id]

            for i in range(self.label_size):
                f_value = 0.
                print("vertex_f_map:"+str(self.vertex_f_map))

                for edge in self.vertex_in_adj_map[vertex_id]:
                    weight = edge.weight
                    src = edge.src
                    deg = self.vertex_deg_map[vertex_id]
                    f_value += self.vertex_f_map[src][i] * (weight / deg)
                next_f_value.append(f_value)
                if self.vertex_label_map[vertex_id] == 0:
                    if f_value > f_values[i]:
                        diff += f_value - f_values[i]
                    else:
                        diff += f_values[i] - f_value
                next_vertex_f_map[vertex_id] = next_f_value

        for vertex_id in self.vertex_label_map.keys():
            if self.vertex_label_map[vertex_id] == 0:
                continue
            next_vertex_f_map[vertex_id] = self.vertex_f_map[vertex_id]

        self.vertex_f_map = next_vertex_f_map

        return diff

    def run(self, eps, max_iter, show_log=False, clean_result=False):
        diff = 0.
        for i in range(max_iter):
            diff = self.iterate()
            if diff < eps:
                break

        if show_log:
            self.show_detail(diff, eps, max_iter)

        ans = self.debug()

        if clean_result:
            rtn_cleaned = []
            for line in ans:
                try:
                    score = sum([float(_[1]) for _ in line[2:]])
                    if score:
                        rtn_cleaned.append([line[0], line[1], score])
                except Exception as e:
                    raise Exception('r')
            ans = rtn_cleaned
        return ans



if __name__ == '__main__':
    labelprop = LabelProp()
    labelprop.load_data_from_file('./data/sample.txt')
    ans = labelprop.run(0.00001, 100, clean_result=True)

    with open('./data/lpop.txt', 'w') as f:
        for line in ans:
            print(str(line))
            f.write((str(line) + '\n'))