# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = '数据预处理.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/9 11:14'
dataBX
"""
import numpy as np

if __name__ == "__main__":

    datapath = lambda at:f"./B题数据/子问题2-数据集B/dataB{at}.csv"




    for i in range(1,2):
        itemsAll = np.loadtxt(datapath(i),  delimiter=',', skiprows=1,  encoding='utf-8')
        print(itemsAll)
