# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = 'v3 单纯演化.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/8 17:31'

本算法直接使用演化, 不做数据处理,
基于以下演化规则:


"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

csvDir = "./B题数据/子问题2-数据集B/dataB3.csv"

t1 = np.loadtxt(csvDir, dtype=np.float_, delimiter=',', skiprows=1, usecols=(3, 4), encoding='utf-8')

plt.scatter()