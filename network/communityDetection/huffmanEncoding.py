import pandas as pd
import numpy as np

#Tree-Node Type

#创建的节点对象
class Node:
    def __init__(self, freq):
        self.left = None
        self.right = None
        self.father = None #符号
        self.freq = freq #权重
    def isLeft(self):
        return self.father.left == self #用于判断它是不是左子树

#create nodes创建叶子节点
def createNodes(freqs):
    return [Node(freq) for freq in freqs]

#create Huffman-Tree创建Huffman树
#把每个节点对象，告诉它，它的左子树节点是哪个，右子树节点是哪个
def createHuffmanTree(nodes):
    queue = nodes[:] #nodes是个List里面存满 nodes对象
    while len(queue)>1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]  #返回的是最后的根节点，里面是个对象，包含了根的左子树和右子树,其他节点虽然没有返回但是节点对象里面已经保存了左子树右子树

#Huffman编码
def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_temp = nodes[i]
        while node_temp != root:
            if node_temp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_temp = node_temp.father
    return codes



if __name__=='__main__':
    chars_freqs = [('C', 2), ('G', 2), ('E', 3), ('K', 3), ('B', 4),
                   ('F', 4), ('I', 4), ('J', 4), ('D', 5), ('H', 6),
                   ('N', 6), ('L', 7), ('M', 9), ('A', 10)]
    freqs = [item[1] for item in chars_freqs]
    nodes = createNodes(freqs)
    root = createHuffmanTree(nodes)
    codes = huffmanEncoding(nodes, root)
    for item in zip(chars_freqs, codes):
        print(item)

    # encoded = ""
    # for s in "abc":
    #     encoded += encode(s, tree)
    # print(encoded)
    # print(decode(encoded, tree))


