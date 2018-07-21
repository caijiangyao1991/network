from SIR import DiseaseNetwork
from SIR import DiseaseNode
from SIR import SIR
import matplotlib.pyplot as plt
import numpy as np


def main(filename):
    dnet = DiseaseNetwork()

    print ("### Loading network ###")
    with open(filename, mode='r') as data_file:
        for line in data_file:
            if line[0] == '#':
                continue
            line = line.replace('\n', '')
            ij_list = line.split('\t')
            i = int(ij_list[0])
            j = int(ij_list[1])
            i_node = dnet.addNode(i)  # type: DiseaseNode
            if i_node is None:
                dnet.getNode(i).addNeighbor(j)
            else:
                i_node.addNeighbor(j)
            dnet.addNode(j)
    print("Network is loaded")

    print ("### SIR ###")
    print ("Initializing SIR model...")
    sir = SIR(beta=0.5, mu=0.2, time_step=0)
    sir.diseaseNet = dnet
    sir.setSeeds(dnet.getRandomNodes(int(dnet.size()*0.01)))
    print ("Starting SIR model...")
    # sir.start()

    __len_susceptibles = []
    __len_recovereds = []
    __len_infecteds = []
    while not sir.isConverged():
        sir.go()
        __len_susceptible = len(sir.diseaseNet.getSusceptibleNodes())
        __len_recovered = len(sir.diseaseNet.getRecoveredNodes())
        __len_infected = len(sir.getInfectedNodes())
        __len_total = __len_susceptible + __len_infected + __len_recovered
        __len_susceptibles.append(__len_susceptible / float(__len_total))
        __len_infecteds.append(__len_infected / float(__len_total))
        __len_recovereds.append(__len_recovered / float(__len_total))
        sir.get_dik_dt(10)

    plt.plot(range(len(__len_susceptibles)), __len_susceptibles, 'y--')
    plt.plot(range(len(__len_recovereds)), __len_recovereds, 'g--')
    plt.plot(range(len(__len_infecteds)), __len_infecteds, 'r--')
    plt.show()
    print("Time :", sir.getTime())

if __name__ == '__main__':
    dataset_filename = "Email-EuAll.txt"
    main(dataset_filename)
