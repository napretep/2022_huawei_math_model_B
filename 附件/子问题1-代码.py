# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = 'v4非智能算法.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/9 5:03'
排序方式:
什么效果? 效果如何? 性能分析?
为什么?
为什么选用最小边长排列? 为什么优先向上堆叠? 为什么最小边长要排序? 为什么要评分选择?
因为通过观察可知, 通常空余部分在上部分,因此上部分尽可能一次填满
由齐头切的规则可知, 如果一个难以拼合的产品放置在板材边缘处,比如x轴上,则其往y轴方向上的可堆叠的产品数量将受到产品的尺寸限制,
因此堆叠的可能性会降低, 因此此方向上的空置率会提升, 为了减少空置率, 即尽可能增加此方向上的占用率, 我们可以选择将产品的较短边作为底, 较长边作为高
有几个优先级
1 先堆底边降序
2 堆高降序

今后的优化方向
1 增加一种栈的并列评分规则
2 动态评分规则

"""
import dataclasses
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import MultipleLocator
import numpy as np
from datetime import datetime
from functools import cmp_to_key
import csv

W = 1220
H = 2440
datapathB = lambda at: f"./B题数据/子问题2-数据集B/dataB{at}.csv"
datapathA = lambda at: f"./B题数据/子问题1-数据集A/dataA{at}.csv"
outputPathA = lambda at: f"./B题数据/子问题1-数据集A/cut_program_dataA{at}.csv"
plt.rcParams["font.sans-serif"]=["SimHei"]

class Constants:
    imgCapacity = 0
    imgRow = 5
    imgCol = 20

class Utils:

    @staticmethod
    def sortCsv():
        """主要将第2题的文件进行按材质分类,"""
        datapath = lambda at: f"./B题数据/子问题2-数据集B/dataB{at}.csv"

        for i in range(5, 6):
            items = Utils.readCsv(datapath(i))
            writer = csv.writer(open(datapath(i)[:-4] + "_sorted.csv", 'w', encoding='UTF8', newline=''))
            writer.writerow(["item_order", "item_material", "item_id", "max_length", "min_length"])
            maxlength = np.array([max(i[0], i[1]) for i in items[:, 3:5].astype(np.float_)])
            minlength = np.array([min(i[0], i[1]) for i in items[:, 3:5].astype(np.float_)])
            itemIds = items[:, 0]
            materials = items[:, 1]
            orders = items[:, 5:6]
            items = np.insert(orders, 1, materials, axis=1)
            items = np.insert(items, 2, itemIds, axis=1)
            items = np.insert(items, 3, maxlength, axis=1)
            items = np.insert(items, 4, minlength, axis=1)
            items = sorted(items, key=cmp_to_key(Cmp.forCsvSort))
            writer.writerows(items)

    @staticmethod
    def groupByOrder(items: "np.ndarray"):  # item_order,item_material,item_id,max_length,min_length
        orderLi = {}
        orderNames = list(set(items[:, 0]))
        for order_id in orderNames:
            orderLi[order_id] = items[[item[0] == order_id for item in items]]
        return orderLi

    @staticmethod
    def print(*args, **kwargs):
        print(f"{datetime.now()}")
        print(*args, **kwargs)

    @staticmethod
    def satisfiesCondition(items: "np.ndarray"):  # 应该只传入长宽数据
        # print(items)
        totalArea = 250 * 1000 * 1000
        totalNum = 1000
        # print(np.dot(items[:,0], items[:,1]) )
        return np.dot(items[:, 0], items[:, 1]) <= totalArea and len(items) <= totalNum

    @staticmethod
    def readCsv(path):
        return np.loadtxt(path, dtype=np.str_, delimiter=',', skiprows=1, encoding='utf-8')

    @staticmethod
    def saveBoardAsCsv(stats: "Statistics", i):
        title = ["原片材质", "原片序号", "产品id", "产品x坐标", "产品y坐标", "产品x方向长度", "产品y方向长度"]
        data = []

        for b in stats.boards:
            data += b.get2DList()

        writer = csv.writer(open(outputPathA(i), 'w', encoding='UTF8', newline=''))
        writer.writerow(title)
        writer.writerows(data)
        pass

    @staticmethod
    def drawBoard(b: "Board"):
        # 4*10, word= series:10,material:Abc,board_id:
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.xlim(0, b.width)
        plt.ylim(0, b.height)

        for boardItemLi in b.stacks:
            for boardItem in boardItemLi:
                ax.add_patch(patches.Rectangle(
                        (boardItem.x, boardItem.y),
                        boardItem.w,
                        boardItem.h,
                        edgecolor='black',
                        facecolor='white',
                        fill=True
                ))
        plt.title(f"{b.id}")


    @staticmethod
    def valueFunc(item: "Item", board: "Board" = None, stack: "list[BoardItem]" = None, stackWeight=2, boardWeight=1) -> "float":
        """
        设置一个价值函数,计算一个零件在一个板材中的价值, 每次都选择最优的放置点.
        优先stack, 其次
        :param item:
        :param board:
        :return:
        """
        stackScore = ((stackWeight * item.minLength / stack[-1].w) if stack is not None else 0)
        boardScore = (boardWeight * (item.minLength * item.maxLength) / board.freeArea()) if stack is not None else boardWeight * (item.minLength * item.maxLength) / (H * W)
        return stackScore + boardScore


class Col:
    @staticmethod
    def map(li: "iter", func: "callable"):
        return list(map(func, li))


class Cmp:
    @staticmethod
    def forItemScore(mayBeMax, curMax):  # [board.id, stack.id, score]
        """第一个参数为下一个元素, 第二个参数为当前最大, 返回1 表示替换最大, 返回-1表示不替换"""
        if mayBeMax[2] == curMax[2]:
            # board
            if mayBeMax[0] == -1:
                return -1
            if curMax[0] == -1:
                return 1
            if mayBeMax[0] == curMax[0]:
                # stack
                if mayBeMax[1] == -1:
                    return -1
                if curMax[1] == -1:
                    return 1
                return curMax[1] - mayBeMax[1]  # 取更小
            return curMax[0] - mayBeMax[0]
        return mayBeMax[2] - curMax[2]

    @staticmethod
    def forCsvSort(_next, _current):  # ["item_order", "item_material", "item_id", "max_length", "min_length"]
        """返回1表示替换,返回-1表示不替换,order->material->min_length"""
        if _next[0] > _current[0]:
            return 1
        elif _next[0] < _current[0]:
            return -1
        else:
            if _next[1] > _current[1]:
                return 1
            elif _next[1] < _current[1]:
                return -1
            else:
                return float(_next[4]) - float(_current[4])
                # if _next[4] > _current[4]:
                #     return 1
                # elif _next[4] < _current[4]:
                #     return -1


class Item:
    def __init__(self, item_id, item_material, item_num, item_length, item_width, item_order):
        self.maxLength = max(float(item_length), float(item_width))
        self.minLength = min(float(item_length), float(item_width))  # 作为条带的基底
        self.id = int(item_id)  # 如果是组合,则用最低的
        self.atBoard = None
        self.order_id = item_order
        self.item_material = item_material

    def switchLength(self):
        minL = self.minLength
        self.minLength = self.maxLength
        self.maxLength = minL

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"(maxlen={self.maxLength},minlen={self.minLength},id={self.id})"


class BoardItem:
    def __init__(self, x, y, w, h, i, item):
        self.x, self.y, self.w, self.h, self.id = x, y, w, h, i
        self.item: "Item" = item

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"x={self.x},y={self.y},w={self.w},h={self.h},id={self.id}\n"

    def getList(self):
        return [self.id, self.x, self.y, self.w, self.h]


class Board:
    def __init__(self, W, H, identity, items: "list[list[BoardItem]]" = None):
        self.stacks: "list[list[BoardItem]]" = items if items is not None else []  # [ [[x, y, width, height,id],[x, y, width, height,id]]]
        self.width = W
        self.height = H
        self.id = identity  # uuid.uuid1().__str__()[:8]

    # def isLegal(self,posi,item:"Item"):

    def createStack(self, item: "Item"):
        if self.atLastStackRight() + item.minLength > self.width:
            raise ValueError("self.atLastStackRight()+item.minLength>self.width")
        self.stacks.append([BoardItem(self.atLastStackRight(), 0, item.minLength, item.maxLength, item.id, item)])
        item.atBoard = self

    def addItemToStack(self, item: "Item", stack: "list[BoardItem]"):
        top = stack[-1]
        stack.append(BoardItem(top.x, top.y + top.h, item.minLength, item.maxLength, item.id, item))
        item.atBoard = self

    def __repr__(self):

        return f"{self.stacks}"

    def getStackHeight(self, stack: "list[BoardItem]"):
        return sum(s.h for s in stack)

    def canStack(self, item: "Item", stack: "list[BoardItem]"):
        stackHeight = sum(s.h for s in stack)
        stackWidth = stack[-1].w
        return item.atBoard is None and stackHeight + item.maxLength <= self.height and item.minLength <= stackWidth

    def canAddNewStack(self, item: "Item"):
        return self.atLastStackRight() + item.minLength <= self.width

    def atLastStackRight(self):
        if len(self.stacks) == 0:
            X = 0
        else:
            X = self.stacks[-1][0].x + self.stacks[-1][0].w
        return X

    def stackArea(self, stack: "list[BoardItem]"):
        return sum(s.h * s.w for s in stack)

    def freeArea(self):
        return self.width * self.height - sum(self.stackArea(stack) for stack in self.stacks)

    def get2DList(self):

        li = []
        for stack in self.stacks:
            # 材质,序号,订单编号,x,y,w,h

            for item in stack:
                # item_id = rawData[item.id, 0]
                # materialName = rawData[item, 1]
                li.append([item.item.item_material, self.id, item.id, item.x, item.y, item.w, item.h])
        return li


class Statistics:
    def __init__(self, data: "np.ndarray"):
        self.boards: "list[Board]" = []
        self.items: "list[list[Item]]" = [[Item(*data[i])] for i in range(len(data))]

    def calcUseRate(self):
        return sum(b.freeArea() for b in self.boards) / (H * W * len(self.boards))

    def freeArea(self):
        return sum([b.freeArea() for b in self.boards])
        pass

    def addNewBoard(self):
        board = Board(W, H, len(self.boards))
        self.boards.append(board)
        return board


# class Stack:
#     """这是一个二维栈,栈顶会有两个,上侧一个,右侧已鞥"""
#
#     def __init__(self, data: "Item", x=0):
#         self.data: "list[list[BoardItem]]" = []
#         self.setBase(x, data)
#
#     def setBase(self, x, item: "Item"):
#         self.data[0] = [BoardItem(x, 0, item.minLength, item.maxLength, item.id, item)]
#
#     def getTops(self):  # 返回两个list, [[上侧栈顶宽度,上侧栈顶可用高度],[右侧栈顶宽,右侧栈顶高]],第二个list可能为空
#         if len(self.data) == 1:
#             base = self.data[0][0]
#             return [[base.w, H - base.h], [0, 0]]
#         else:
#             # top
#             topItem = self.data[-1][0]
#             topAvailaleHeight = H - (topItem.y + topItem.h)
#             topAvailaleWidth = sum(item.w for item in self.data[-1])
#             # right
#             rightItem = self.data[-1][-1]
#             rightAvailaleHeight = rightItem.h
#             rightAvailaleWidth = sum(item.w for item in self.data[-2]) - sum(item.w for item in self.data[-1])
#             return [[topAvailaleWidth, topAvailaleHeight], [rightAvailaleWidth, rightAvailaleHeight]]
#
#     def pushToRight(self, item: "Item"):
#         x = self.data[-1][-1].x + self.data[-1][-1].w
#         y = self.data[-1][-1].y
#         self.data[-1].append(BoardItem(x, y, item.minLength, item.maxLength, item.id, item))
#
#     def pushToTop(self, item: "Item"):
#         x = self.data[-1][0].x
#         y = self.data[-1][0].y + self.data[-1][0].h
#         self.data[-1].append(BoardItem(x, y, item.minLength, item.maxLength, item.id, item))
#
#     def canPush(self,item):

@dataclasses.dataclass
class Score:
    item: "Item"
    board_id: int  # 如果为负, 则表示新建board
    stack_id: list[int]  # 如果为空, 则表示新建stack
    stackScore: float
    boardScore: float
    rotate: bool = False
    stackWeight: "Union[int,float]" = 2
    boardWeight: "Union[int,float]" = 1

    @property
    def score(self):
        return self.stackScore * self.stackWeight + self.boardScore * self.boardWeight


def v2_run(stats: "Statistics"):
    """加入评分机制来选择板块

    items 为n行2列矩阵
    """
    itemsli = stats.items
    itemsli.sort(key=lambda x: x[0].minLength)
    valueFunc = Utils.valueFunc
    for i in range(len(itemsli)-1,-1,-1):
        item = itemsli[i][0]
        if item.atBoard is not None: continue
        # 开始评分
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

        # 开始选取最高分
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
    for i in range(1, 6):
        items = Utils.readCsv(datapathA(i))
        stats = Statistics(items)
        v2_run(stats)
        Utils.print(1 - stats.freeArea() / (len(stats.boards) * (H * W)))
        Utils.print(len(stats.boards))

        Utils.saveBoardAsCsv(stats, i)
        # writer = csv.writer(open(outputPathA(i), 'w', encoding='UTF8', newline=''))
        # writer.writerow(["批次序号","原片材质","原片序号","产品id","产品x坐标","产品y坐标","产品x方向长度","产品y方向长度"])
        # writer.writerows(allTable)
        #
        # Constants.imgCapacity=0
        # fig = plt.figure(figsize=(34, 12))
        # for b in stats.boards:
        #     plt.subplot(Constants.imgRow, Constants.imgCol, (Constants.imgCapacity % (Constants.imgRow * Constants.imgCol)) + 1)
        #     Utils.drawBoard(b)
        #     Constants.imgCapacity += 1
        #     if Constants.imgCapacity % (Constants.imgRow * Constants.imgCol) == 0:
        #         plt.savefig(f"dataA{i}_{int(Constants.imgCapacity / (Constants.imgRow * Constants.imgCol))}.png", bbox_inches='tight')
        #         plt.clf()
        # if Constants.imgCapacity % (Constants.imgRow * Constants.imgCol) != 0:
        #     plt.savefig(f"dataA{i}_{int(Constants.imgCapacity/(Constants.imgRow*Constants.imgCol))+1}.png",bbox_inches='tight')
        #     plt.clf()
    pass
