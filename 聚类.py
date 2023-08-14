# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = '聚类.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/7 17:49'
"""

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from itertools import cycle
from sklearn.datasets import make_blobs

if __name__ == "__main__":

    # 产生的数据个数
    n_samples = 3000
    # 数据中心点
    centers = [[2, 2], [-1, -1], [1, -2]]
    # 生产数据
    X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1, random_state=0)
    print(X[True,0])

    # variables = ['X', 'Y']
    # # 层次聚类树
    # df = pd.DataFrame(X, columns=variables, index=labels)
    # '''
    # pdist：计算样本距离,其中参数metric代表样本距离计算方法
    # (euclidean:欧式距离
    # minkowski:明氏距离
    # chebyshev:切比雪夫距离
    # canberra:堪培拉距离)
    # linkage：聚类,其中参数method代表簇间相似度计算方法
    # (single:  MIN
    # ward：沃德方差最小化
    # average：UPGMA
    # complete：MAX)
    # '''
    # row_clusters = linkage(pdist(df, metric='canberra'), method='complete')
    # print(pd.DataFrame(row_clusters, columns=['row label1', 'row label2', 'distance', 'no. of stacks in clust.'],
    #                    index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]))
    #
    # # 绘图 层次聚类树
    # # row_dendr = dendrogram(row_clusters, labels=labels)
    # # plt.tight_layout()
    # # plt.title('canberra-complete')
    # # plt.show()
    #
    # # 凝聚层次聚类，应用对层次聚类树剪枝
    # n_clusters_ = 3  # 分三类
    # '''
    # 使用Agglomerative Hierarchical Clustering算法
    # 其中参数affinity代表样本距离计算方法  参数linkage代表簇间相似度计算方法
    # '''
    # ac = AgglomerativeClustering(n_clusters=n_clusters_, affinity='canberra', linkage='complete')
    # labels = ac.fit_predict(X)
    # print('cluster labels:%s' % labels)
    #
    # # 绘图
    # plt.figure(1)
    # plt.clf()
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(0, 3), colors):
    #     ##根据lables中的值是否等于k，重新组成一个True、False的数组
    #     my_members = labels == k
    #     ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    #     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    # plt.title('canberra-complete')
    # plt.show()