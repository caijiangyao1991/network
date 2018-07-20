from py2neo import Graph,Node,Relationship  #Neo4j 的Python驱动库py2neo
from igraph import Graph as IGraph

graph = Graph("http://localhost:7474", username="neo4j", password="q6594461")
query =  "Match (n1:firm)-[r]-(n2:firm) return n1.name, n2.name, type(r) as rel"
#直接传入Py2neo查询结果对象到igraph的TupleList构造器
ig = IGraph.TupleList(graph.run(query), weights=False)


# 使用igraph实现的随机游走算法
clusters = IGraph.community_walktrap(ig, weights=None).as_clustering()
nodes = [{"name": node["name"]} for node in ig.vs]
for node in nodes:
    idx = ig.vs.find(name=node["name"]).index #找到节点的index
    node["community"] = clusters.membership[idx] #根据index返回cluster群
print(nodes)

