#encoding = utf-8
import numpy as np
import networkx as nx
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def createGraph(filename,firstline):
    G = nx.DiGraph()
    f = open(file,'r',encoding="utf-8")
    for line in f.readlines():
        if firstline == True:
            firstline = False
            continue
        strlist = line.replace('\n','').split(",")
        n1 = strlist[0]
        n2 = strlist[1]
        r = strlist[2]
        try:
            w = int(strlist[3])
            G.add_weighted_edges_from([(n1, n2, w)])
            G.add_edges_from([(n1, n2, {'relationship':r})]) #添加关系
        except Exception as e:
            print(e)
    return G

def find_communities(G,T):
    """
    :param G: graph
    :param T: iteration
    :param r: threshold
    :return: community
    """
    #TODO stage1
    #将图中数据录入到数据字典中以便使用
    weight = {j:{} for j in G.nodes()}
    for q in weight.keys():
        for m in G[q].keys():
            weight[q][m] = G[q][m]['weight']
    #建立成员标签记录
    memory = {i:{i:1} for i in G.nodes()}

    #TODO stage2
    #开始遍历T次所有节点
    for t in range(T):
        listenerslist = list(G.nodes())
        #随机排列遍历顺序
        np.random.shuffle(listenerslist)
        #开始遍历节点
        for listener in listenerslist:
            # 每个节点的key就是与他相连的节点标签名
            speakerlist = G[listener].keys()
            if len(speakerlist) == 0:
                continue
            labels = defaultdict(int)
            #遍历所有与其相关联的节点
            for j,speaker in enumerate(speakerlist):
                total = float(sum(memory[speaker].values()))
                # 查看speaker中memory中出现概率最大的标签并记录(根据概率进行多项分布的抽样，抽出后取其中值最大的对应的index)
                # key是标签名，value是Listener与speaker之间的权
                #先是选出每个spear的标签
                labels[list(memory[speaker].keys())[
                    np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += \
                weight[listener][speaker]
            #再是根据speaker的标签
            #Listener Rule查看label中值最大的标签，让其成为当前listener的一个记录
            maxlabel = max(labels, key=labels.get)
            #update listener memory
            if maxlabel in memory[listener]:
                memory[listener][maxlabel] += 1
            else:
                memory[listener][maxlabel] = 1

    # TODO stage3
    # 提取出每个节点memory中记录标签出现最多的一个
    for primary in memory:
        p = list(memory[primary].keys())[
                    np.random.multinomial(1, [freq / total for freq in memory[primary].values()]).argmax()]
        memory[primary] = {p: memory[primary][p]}
    communities = {}
    #扫描memory中的记录标签，相同标签的节点加入同一个社区中
    for primary, change in memory.items():
        for label in change.keys():
            if label in communities:
                communities[label].add(primary)
            else:
                communities[label] = set([primary])
    return communities

def resultDf(communities,G):
    nodes = [{"name": node} for node in G.nodes]
    for node in nodes:
        for key, value in communities.items():
            if node["name"] in value:
                node["community"] = key
    result = pd.DataFrame(nodes)
    class_mapping = {label: idx for idx, label in enumerate(set(result['community']))}
    result['community'] = result['community'].map(class_mapping)
    return result

if __name__=="__main__":
    file = "./data/load.csv"
    G = createGraph(file,True)
    communities = find_communities(G, 100)
    print(communities)
    result = resultDf(communities,G)
    print(result.head())
    result.to_csv('./outData/commDetect_SLPA.csv', encoding='gb18030', index=False)











