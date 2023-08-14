from typing import Callable

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from itertools import cycle
from sklearn.datasets import make_blobs



csvDir =lambda d: f"./B题数据/子问题1-数据集A/dataA{d}.csv"

def clust(CSVDir,n_clusters_):
    data_1 = np.loadtxt(CSVDir, dtype=np.float_, delimiter=',', skiprows=1, usecols=(3, 4), encoding='utf-8') #DataFrame(c)
    d3_l = []
    d3_sq = []
    items= [[max(*data_1[i]),data_1[i][0]*data_1[i][1]] for i in range(len(data_1))]

    ac = AgglomerativeClustering(n_clusters=n_clusters_, affinity='canberra', linkage='complete')
    labels = ac.fit_predict(items)
    # stacks = np.insert(stacks, 2, labels, axis=1)
    returnData=[]
    for i in range(n_clusters_):
        returnData.append(([items[j] for j in range(len(items)) if i==labels[j]]))
    # print(np.array(returnData)[1])
    print([len(i) for i in returnData])
    return returnData


def plot_all(data):
    """
    [
        [ dataset
            [ label 1
                [length,area,label]
                [length,area,label]
                ...
            ],
            [ label 2
                [length,area,label]
                [length,area,label]
                ...
            ],
        ]
        ...
    ]

    :param data:
    :return:
    """


    for datasetnum in range(len(data)):
        dataset = data[datasetnum]
        for labelnum in range(len(dataset)):
            label = dataset[labelnum]
            fig=plt.figure(figsize=(20,10))
            ax = fig.add_subplot(1, labelnum+1, labelnum + 1)
            nplabel = np.array(label)
            ax.scatter(nplabel[:,0],nplabel[:,1],s=1)
            ax.set_title(f"s")
            ax.legend([str(i) for i in range(len(data))])

if __name__ == "__main__":
    data = []
    # data.append(clust(csvDir(1), 7))
    for i in range(1,6):
        data.append(clust(csvDir(i),7))
    # print(data[:10])
    plot_all(data)
    plt.show()
    pass