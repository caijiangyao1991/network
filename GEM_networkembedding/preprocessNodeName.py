# encoding = utf-8
import numpy as np
import pandas as pd

data = pd.read_csv("./data/load.csv")
source = list(data['companyA'])
verter = list(data['companyB'])
source.extend(verter)
node = list(set(source))
node_dict = dict(zip(node,range(len(node))))
data['source']=data['companyA'].replace(node_dict)
data['target']=data['companyB'].replace(node_dict)
data1 = data[['source','target','weight']]
data1.to_csv('./data/load_rename.csv',index=False,header=None)