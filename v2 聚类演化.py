# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = 'v2 聚类演化.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/7 16:55'

算法思路
1 读取csv文件
2 对csv文件预处理:
    2.1 得到列表, 整理其元素为 [max边长,min边长, 面积,id]
    2.2 对列表统计各类max/min相同的频数, 包括 min中有max 和 max中有min的频数
    2.3 排序2.2用于统计的key, 先根据频数排序,再根据宽度排序,用于下一阶段的长尾填充
    2.4 对频数为1的进行聚类分析, 不极端的排在一起, 极端(太胖,太瘦)的排在一起.
    2.5 设定一个频数门槛, 低于频数门槛的把它们组合成单个item看待(比如根据中位数来设计,不能组合出极端的), 也视为统计频数, 再更新统计频数,
    2.6 循环 2.4与2.5 直到所有频数门槛以下的item都被转为组合频数为1的对象, 实在无法组合的就放回2,3频数
3 循环迭代:
    3.1 先处理长尾中的不极端数据, 再处理极端数据中的大头,
    3.1.1 等大头和不极端数据处理完成后, 把极端长条插入, 再从频数门槛以上的选item插入, 直到无法插入为止.
    3.2 长尾部分处理完后, 用演化算法处理剩余部分, 让他们针对一个板材组合出最优方案, 主要是对频数门槛以上的,
    3.2.2 最后将剩余的频数门槛以上的, 使用遗传算法处理出最优值, 遗传算法的参数下次再考虑
需要设置一个函数判断如果插入某个item到指定位置, 还能否保持三阶段完成性质.
"""
import time
from typing import Optional
import uuid

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from functools import cmp_to_key, reduce
from matplotlib.pyplot import MultipleLocator
n_clusters_ = 2
W = 1220
H = 2440

ac = AgglomerativeClustering(n_clusters=n_clusters_, affinity='canberra', linkage='complete')


class Utils:
    @staticmethod
    def print(*args, **kwargs):
        print(f"{datetime.now()}")
        print(*args, **kwargs)


class Col:
    @staticmethod
    def map(li: "iter", func: "callable"):
        return list(map(func, li))


def clustering(items_):
    ac = AgglomerativeClustering(n_clusters=n_clusters_, affinity='canberra', linkage='complete')
    labels = ac.fit_predict(items_)


class Cmp:
    @staticmethod
    def forItemScore(mayBeMax, curMax): # [board.id, stack.id, score]
        """第一个参数为下一个元素, 第二个参数为当前最大, 返回1 表示替换最大, 返回-1表示不替换"""
        if mayBeMax[2]==curMax[2]:
            # board
            if mayBeMax[0] == -1:
                return -1
            if curMax[0] == -1:
                return 1
            if mayBeMax[0]==curMax[0]:
                # stack
                if mayBeMax[1]==-1:
                    return -1
                if curMax[1]==-1:
                    return 1
                return curMax[1]-mayBeMax[1] # 取更小
            return curMax[0]-mayBeMax[0]
        return mayBeMax[2] - curMax[2]
def cmpForMaxMin(_next, _prev):
    if _next[1] < _prev[1]:
        return 1
    elif _next[1] > _prev[1]:
        return -1
    else:
        if _next[0] >= _prev[0]:
            return 1
        else:
            return -1


def funStatistic(items_):
    """
    cluster {
        "normal":[
            "width1":[]
            "width2":[]
                ...
        ],
        "tooFat":[
            "width1":[]
            "width2":[]
                ...
        ],
        "tooThin":[
            "width1":[]
            "width2":[]
                ...
        ]
    }

    :param stacks:
    :return:
    [
        [ label1
            [lengt, width, label, idh]  [lengt, width, label, idh]  [lengt, width, label, idh]  [lengt, width, label, idh]
        ]
        [ label2
            [lengt, width, label, idh]  [lengt, width, label, idh]  [lengt, width, label, idh]  [lengt, width, label, idh]
        ]
        ...
    ]
    """
    items = [[max(*items_[i]), items_[i][0] * items_[i][1]] for i in range(len(items_))]

    d3_l = []
    d3_sq = []

    ac = AgglomerativeClustering(n_clusters=n_clusters_, affinity='canberra', linkage='complete')
    labels = ac.fit_predict(items)
    # stacks = np.insert(stacks, 2, labels, axis=1)
    returnData = []
    items = [items_[i] + labels[i] + [i] for i in range(len(labels))]
    for i in range(n_clusters_):  # 根据label分类
        returnData.append(([items[j] for j in range(len(items)) if i == labels[j]]))
    # print(np.array(returnData)[1])
    print([len(i) for i in returnData])
    return returnData
    # 每个元素 都是一个聚类分支, 代表了一类零件的形状特征, 比如比较均匀的 可以用遗传算法优化聚类的标准参数
    # 一般分5个区,


class Stripe:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.items = []  # 从底到高
        self.position = [0, 0]  # 位置
        self.plate = None


class Item:
    def __init__(self, maxLength=0, minLength=0, identity=-1):
        self.maxLength = maxLength
        self.minLength = minLength  # 作为条带的基底
        self.id = identity  # 如果是组合,则用最低的
        self.ids = []
        self.atBoard = None
        self.label = -1
        self.boardScore:float = .0 # str 为 board 的 id , float为 这个 Item在此处的得分, item总是在得分最高的board中
    def switchLength(self):
        minL = self.minLength
        self.minLength = self.maxLength
        self.maxLength = minL

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"(maxlen={self.maxLength},minlen={self.minLength},id={self.id})"


class BoardItem:
    def __init__(self,x,y,w,h,i):
        self.x,self.y,self.w,self.h,self.id = x,y,w,h,i
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f"x={self.x},y={self.y},w={self.w},h={self.h},id={self.id}\n"

class Board:
    def __init__(self,W,H,identity,items:"list[list[BoardItem]]" = None):
        self.stacks: "list[list[BoardItem]]" = items if items is not None else [] #[ [[x, y, width, height,id],[x, y, width, height,id]]]
        self.width = W
        self.height= H
        self.id = identity # uuid.uuid1().__str__()[:8]
    # def isLegal(self,posi,item:"Item"):

    def createStack(self, item:"Item"):
        if self.atLastStackRight()+item.minLength>self.width:
            raise ValueError("self.atLastStackRight()+item.minLength>self.width")
        self.stacks.append([BoardItem(self.atLastStackRight(), 0, item.minLength, item.maxLength, item.id)])
        item.atBoard = self

    def addItemToStack(self,item:"Item",stack:"list[BoardItem]"):
        top = stack[-1]
        stack.append(BoardItem(top.x,top.y+top.h,item.minLength,item.maxLength,item.id))
        item.atBoard = self

    def __repr__(self):

        return f"{self.stacks}"

    def getStackHeight(self, stack:"list[BoardItem]"):
        return sum(s.h for s in stack)

    def canStack(self,item:"Item",stack:"list[BoardItem]"):
        stackHeight = sum(s.h for s in stack)
        stackWidth = stack[-1].w
        return item.atBoard is None and stackHeight+item.maxLength<=self.height and item.minLength<=stackWidth

    def canAddNewStack(self,item:"Item"):
        return self.atLastStackRight() + item.minLength <= self.width

    def atLastStackRight(self):
        if len(self.stacks)==0 : X = 0
        else:
            X = self.stacks[-1][0].x + self.stacks[-1][0].w
        return X

    def freeArea(self):
        return self.width*self.height - sum(self.getStackHeight(stack)*stack[-1].w for stack in self.stacks)

    # def insertable(self, x,y, item: "Item"):
    #     x+item.minLength < self.width

class Statistics:
    def __init__(self,data):
        self.items: "list[list[Item]]" = []
        self.boards :"list[Board]"= []
        self.items: "list[list[Item]]" = [[Item(max(*data[i]), min(*data[i]), i)] for i in range(len(data))]
        self.originItems = [(max(*data[i]), min(*data[i]), i) for i in range(len(data))]
        # self.boardsDict:"dict[]"
        self.originItems: "list[tuple[float,float,int]]" = []
        self.composedItems: "list[list[int]]" = []
        self.max: "dict[int,list[list[Item]]]" = {}
        self.min: "dict[int,list[list[Item]]]" = {}  # { minLen1:[[item],[item]] }
        self.minCanNotCompose = []
        self.maxCanNotCompose = []

        self.minCount = []  # [[minLen,count],[minLen,count]]
        self.maxCount = []

    def addNewBoard(self):
        board = Board(W,H,len(self.boards))
        self.boards.append(board)
        return board

    @staticmethod
    def getSize(items: "list[Item]") -> "tuple[float,float]":

        return items[0].minLength, sum(i.maxLength for i in items)

    def getMaxLenAndArea(self, size: "tuple[float,float]" = None, items: "list[Item]" = None):
        if size:
            return size[0], size[1] * size[0]
        if items:
            return self.getMaxLenAndArea(self.getSize(items))

    def maxMinGen(self, data):


        for i in range(len(self.items)):
            maxlen = self.items[i][0].maxLength
            minlen = self.items[i][0].minLength
            if maxlen not in self.max:
                self.max[maxlen] = []
            if minlen not in self.min:
                self.min[minlen] = []
            self.max[maxlen].append(self.items[i])
            self.min[minlen].append(self.items[i])

    def maxMinCount(self):
        self.maxCount = [[key, len(self.max[key])] for key in self.max.keys()]
        self.minCount = [[key, len(self.min[key])] for key in self.min.keys()]
        self.maxCount.sort(key=cmp_to_key(cmpForMaxMin))
        self.minCount.sort(key=cmp_to_key(cmpForMaxMin))


    def getMinCountByFreq(self, count) -> "list[float]":
        return [key[0] for key in self.minCount if key[1] == count and key[0] not in self.minCanNotCompose]


drawedRect =[]

def drawBoard(b:"Board"):
    fig = plt.figure(figsize=(4,8))

    ax = plt.gca()
    # plt.pause(0.1)
    # fig, ax = plt.subplots()
    # ax.plot([], [])
    x_major_locator = MultipleLocator(100)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(200)
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlim(0, b.width)
    plt.ylim(0, b.height)
    for boardItemLi in b.stacks:
        for boardItem in boardItemLi:
            ax.add_patch(patches.Rectangle(
                            (boardItem.x,boardItem.y),
                            boardItem.w,
                            boardItem.h,
                            edgecolor='blue',
                            facecolor='green',
                            fill=True
                    ))

    fig.savefig(f"{datetime.now()}.png".replace(":","-"))


def dataMinMax(data):
    stats = Statistics()
    # Utils.print("minmax 函数开始")

    stats.maxMinGen(data)

    stats.maxMinCount()

    # Utils.print("minmax 函数结束")


    return stats


def calcCutPosition(board:"Board",items:"list[list[Item]]",base =False, debug = False):
    """
    优先处理 栈堆叠
    :param board:
    :param items:
    :param base:
    :param debug:
    :return:
    """
    for i in range(len(items) - 1, -1, -1):
        item = items[i][0]
        if item.atBoard is not None : continue
        if len(board.stacks)>0:
            for stack in board.stacks:
                # topItem: "BoardItem" = stack[-1]
                # Utils.print(stack)
                if board.canStack(item,stack): #先往上叠,
                    board.addItemToStack(item, stack)
                    break

        else:
            board.createStack(item)


def valueFunc(item:"Item",board:"Board"=None,stack:"list[BoardItem]"=None, stackWeight=2, boardWeight=1)->"float":
    """
    设置一个价值函数,计算一个零件在一个板材中的价值, 每次都选择最优的放置点.
    优先stack, 其次
    :param item:
    :param board:
    :return:
    """
    stackScore = ((stackWeight * item.minLength/stack[-1].w) if stack is not None else 0)
    boardScore = (boardWeight*(item.minLength*item.maxLength)/board.freeArea()) if stack is not None else boardWeight*(item.minLength*item.maxLength)/(H*W)
    return stackScore+boardScore

def v1_run(items):
    allPlate = []
    stats = dataMinMax(items)
    Utils.print(stats.minCount)
    freqLi:"list[int]" = sorted(list(set(i[1] for i in stats.minCount)))
    Utils.print(f"freqLi={freqLi}")

    # for itemW  in sorted(stats.min.keys()):


    Utils.print("开始主循环1")
    nowX, nowY = 0, 0
    whenKeepChange = True
    for f in freqLi:
        itemsli: "list[list[Item]]" = reduce(lambda x, y: x + y, (stats.min[i] for i in stats.getMinCountByFreq(f)))
    # itemsli = stats.items
        itemsli.sort(key=lambda x: x[0].maxLength)
        for i in range(len(itemsli) - 1, -1, -1):
            item = itemsli[i][0]
            if item.atBoard is not None: continue

            if len(stats.boards)==0:stats.addNewBoard()
            for board in stats.boards:
                if item.atBoard is not None: break
                if len(board.stacks)==0:board.createStack(item)
                for stack in board.stacks:
                    if board.canStack(item, stack):  # 先往上叠,
                        board.addItemToStack(item, stack)
                        break
                if item.atBoard is not None: break
                if board.atLastStackRight() + item.minLength <= board.width:  # 叠不住了再创建新栈
                    board.createStack(item)
                    break
            if item.atBoard is None:
                board = stats.addNewBoard()
                board.createStack(item)
                continue
    Utils.print(len(stats.boards))
    Utils.print("结束主循环")

def v2_run(items):
    """加入评分机制来选择板块"""
    allPlate = []
    stats = dataMinMax(items)
    # Utils.print(stats.minCount)
    freqLi:"list[int]" = sorted(list(set(i[1] for i in stats.minCount)))
    # Utils.print(f"freqLi={freqLi}")

    # for itemW  in sorted(stats.min.keys()):


    # Utils.print("开始主循环1")
        # itemsli: "list[list[Item]]" = reduce(lambda x, y: x + y, (stats.min[i] for i in stats.getMinCountByFreq(f)))
    itemsli = stats.items
    itemsli.sort(key=lambda x: x[0].minLength)

    for i in range(len(itemsli) - 1, -1, -1):
        item = itemsli[i][0]
        if item.atBoard is not None: continue

        # if len(stats.boards)==0:stats.addNewBoard()
        # 评分阶段

        scoreLi = [[-1,-1,valueFunc(item),False]]
        for board in stats.boards:
            for s in range(len(board.stacks)): # 如果有stack 则对不同的stack放置打分
                if board.canStack(item,board.stacks[s]):
                    scoreLi.append([board.id,s,valueFunc(item,board,board.stacks[s]),False])
            if board.canAddNewStack(item): # 如果可以作为新stack的base, 则打分
                scoreLi.append([board.id,-1,valueFunc(item,board),False])
        item.switchLength()
        scoreLi += [[-1,-1,valueFunc(item),True]] # TODO 交换长宽需要作为排序依据
        for board in stats.boards:
            for s in range(len(board.stacks)): # 如果有stack 则对不同的stack放置打分
                if board.canStack(item,board.stacks[s]):
                    scoreLi.append([board.id,s,valueFunc(item,board,board.stacks[s]),True])
            if board.canAddNewStack(item): # 如果可以作为新stack的base, 则打分
                scoreLi.append([board.id,-1,valueFunc(item,board),True])
        item.switchLength()
        higherScore = max(scoreLi,key=cmp_to_key(Cmp.forItemScore))

        if higherScore[3]:item.switchLength() # 如果需要转置矩形
        if higherScore[0]==-1:
            board = stats.addNewBoard()
            board.createStack(item)
        elif higherScore[1]==-1:
            board = stats.boards[higherScore[0]]
            board.createStack(item)
        else:
            board = stats.boards[higherScore[0]]
            board.addItemToStack(item,board.stacks[higherScore[1]])

    # for b in stats.boards:
    #     drawBoard(b)
        # Utils.print(b)
    return stats
    # Utils.print("结束主循环")

def v3_run(items):
    """这个函数是失败的"""
    allPlate = []
    stats = dataMinMax(items)
    # Utils.print(stats.minCount)
    freqLi: "list[int]" = sorted(list(set(x[1] for x in stats.minCount)))
    # Utils.print(f"freqLi={freqLi}")

    # for itemW  in sorted(stats.min.keys()):
    for f in freqLi:
        # Utils.print("开始主循环1")
        # itemsli: "list[list[Item]]" = reduce(lambda x, y: x + y, (stats.min[k] for k in stats.getMinCountByFreq(f)))
        itemsli = stats.items
        itemsli.sort(key=lambda x: x[0].minLength)

        for i in range(len(itemsli) - 1, -1, -1):
            item = itemsli[i][0]
            if item.atBoard is not None: continue

            # if len(stats.boards)==0:stats.addNewBoard()
            # 评分阶段

            scoreLi = [[-1, -1, valueFunc(item), False]]
            for board in stats.boards:
                for s in range(len(board.stacks)):  # 如果有stack 则对不同的stack放置打分
                    if board.canStack(item, board.stacks[s]):
                        scoreLi.append([board.id, s, valueFunc(item, board, board.stacks[s]), False])
                if board.canAddNewStack(item):  # 如果可以作为新stack的base, 则打分
                    scoreLi.append([board.id, -1, valueFunc(item, board), False])
            item.switchLength()
            scoreLi += [[-1, -1, valueFunc(item), True]]  # TODO 交换长宽需要作为排序依据
            for board in stats.boards:
                for s in range(len(board.stacks)):  # 如果有stack 则对不同的stack放置打分
                    if board.canStack(item, board.stacks[s]):
                        scoreLi.append([board.id, s, valueFunc(item, board, board.stacks[s]), True])
                if board.canAddNewStack(item):  # 如果可以作为新stack的base, 则打分
                    scoreLi.append([board.id, -1, valueFunc(item, board), True])
            item.switchLength()
            higherScore = max(scoreLi, key=cmp_to_key(Cmp.forItemScore))

            if higherScore[3]: item.switchLength()  # 如果需要转置矩形
            if higherScore[0] == -1:
                board = stats.addNewBoard()
                board.createStack(item)
            elif higherScore[1] == -1:
                board = stats.boards[higherScore[0]]
                board.createStack(item)
            else:
                board = stats.boards[higherScore[0]]
                board.addItemToStack(item, board.stacks[higherScore[1]])
    return stats

if __name__ == "__main__":
    for i in range(1,5):
        items = np.loadtxt(f"./B题数据/子问题1-数据集A/dataA{i}.csv", dtype=np.float_, delimiter=',', skiprows=1, usecols=(3, 4), encoding='utf-8')
        s=v2_run(items)
        Utils.print(f"dataA{i} need boards:{len(s.boards)}")

    pass
