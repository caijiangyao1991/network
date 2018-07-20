#coding=utf8
"""标准化互信息，可以自己写也可以直接调用scikit-learn包中集成的度量函数"""
import numpy as np
import math

def NMI(A, B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
        # 标准化互信息
        Hx = 0
        for idA in A_ids:
            idAOccurCount = 1.0 * len(np.where(A == idA)[0])
            Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
        Hy = 0
        for idB in B_ids:
            idBOccurCount = 1.0 * len(np.where(B == idB)[0])
            Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
        MIhat = 2.0 * MI / (Hx + Hy)
        return MIhat


if __name__=='__main__':
    A = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    B = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])
    print (NMI(A,B))