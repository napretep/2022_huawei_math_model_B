# -*- coding: utf-8 -*-
"""
这个版本会尝试2维栈的效果.
"""
import dataclasses
import uuid
from typing import Optional, Union
from scipy.stats import norm,powerlaw,zipf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import MultipleLocator
import numpy as np
from datetime import datetime
from functools import cmp_to_key
import csv


W = 1220
H = 2440
datapathB = lambda at: f"./B题数据/子问题2-数据集B/dataB{at}_sorted.csv"
datapathA = lambda at: f"./B题数据/子问题1-数据集A/dataA{at}.csv"

class Constants:
    top = 0
    right = 1
    minimum10E_4 = 10e-4  # 0.001
    minimum10E_5 = 10e-5  # 0.0001
    imgCapacity = 0
    imgRow = 5
    imgCol = 20
    random=3

class Utils:
    @staticmethod
    def drawStats(stats:"Statistics",savename=""):
        Constants.imgCapacity = 0
        fig = plt.figure(figsize=(34, 12))
        for b in stats.boards:
            plt.subplot(Constants.imgRow, Constants.imgCol, (Constants.imgCapacity % (Constants.imgRow * Constants.imgCol)) + 1)
            Utils.drawBoard(b)
            Constants.imgCapacity += 1
            if Constants.imgCapacity % (Constants.imgRow * Constants.imgCol) == 0:
                plt.savefig(f"{savename}{i}_{int(Constants.imgCapacity / (Constants.imgRow * Constants.imgCol))}.png",bbox_inches='tight')
                plt.clf()
        if Constants.imgCapacity % (Constants.imgRow * Constants.imgCol) != 0:
            plt.savefig(f"{savename}{i}_{int(Constants.imgCapacity/(Constants.imgRow*Constants.imgCol))+1}.png",bbox_inches='tight')
            plt.clf()
    @staticmethod
    def Zipf(a: np.float64, min: np.uint64, max: np.uint64, size=None):
        """
        Generate Zipf-like random variables,
        but in inclusive [min...max] interval
        """
        if min == 0:
            raise ZeroDivisionError("")

        v = np.arange(min, max + 1)  # values to sample
        p = 1.0 / np.power(v, a)  # probabilities
        p /= np.sum(p)  # normalized

        return np.random.choice(v, size=size, replace=True, p=p)

    @staticmethod
    def saveBoardAsCsv(table, csvFrom: "str", items: "np.ndarray",Q=1):
        writer = csv.writer(open(f"{csvFrom[:-4]}_cut_program.csv", 'w', encoding='UTF8', newline=''))
        if Q==1:
            title = ["原片材质", "原片序号", "产品id", "产品x坐标", "产品y坐标", "产品x方向长度", "产品y方向长度"]

        elif Q==2:
            title = ["批次序号", "原片材质", "原片序号", "产品id", "产品x坐标", "产品y坐标", "产品x方向长度", "产品y方向长度"]


        writer.writerow(title)
        writer.writerows(table)

        pass

    @staticmethod
    def log(row,path="log.csv"):
        writer= csv.writer(open(path, 'a', encoding='UTF8', newline=''))
        writer.writerow([f"\"{datetime.now()}\"",*row])


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
    def drawBoard(b: "Board", word=""):

        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.xlim(0, W)
        plt.ylim(0, H)

        for stack in b.stacks:
            for row in stack.data:
                for boardItem in row:
                    ax.add_patch(patches.Rectangle(
                            (boardItem.x, boardItem.y),
                            boardItem.w,
                            boardItem.h,
                            edgecolor='blue',
                            facecolor='white',
                            fill=True
                    ))

        m = b.stacks[0].data[0][0].item.material_id
        s = b.stacks[0].data[0][0].item.seires_id

        plt.title(f"{s}_{m}_{b.id}")

class Item:
    def __init__(self,data,Q=1 ):
        if Q==1:#item_id, item_material, item_num, item_length, item_width, item_order)
            self.maxLength = max(float(data[3]), float(data[4]))
            self.minLength = min(float(data[3]), float(data[4]))  # 作为条带的基底
            self.id = int(data[0])  # 如果是组合,则用最低的
            self.order_id = data[5]
            self.material_id = data[1]
            self.seires_id = ""
        if Q==2:#P,item_order,item_material,item_id,max_length,min_length
            self.maxLength = float(data[4])
            self.minLength = float(data[5])  # 作为条带的基底
            self.id = data[3]  # 如果是组合,则用最低的
            self.order_id = data[1]
            self.material_id = data[2]
            self.seires_id = data[0]
        self.atBoard: "Optional[Board]" = None

    def switchLength(self):
        minL = self.minLength
        self.minLength = self.maxLength
        self.maxLength = minL

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"(maxlen={self.maxLength},minlen={self.minLength},id={self.id})"

    def getArea(self):
        return self.maxLength * self.minLength





class Q2:
    @staticmethod
    def groupByOrder(items: "np.ndarray"):  # item_order,item_material,item_id,max_length,min_length
        orderLi = {}
        orderNames = list(set(items[:, 0]))
        for order_id in orderNames:
            orderLi[order_id] = items[[item[0] == order_id for item in items]]
        return orderLi

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


class Col:
    @staticmethod
    def map(li: "iter", func: "callable"):
        return list(map(func, li))


class Cmp:
    @staticmethod
    def forItemSort(_next:"Item",_prev:"Item"):
        if _next.minLength==_prev.minLength:
            return int(-(_next.maxLength - _prev.maxLength))
        else:
            return int(-(_next.minLength - _prev.minLength))

    @staticmethod
    def forItemScore2(mayBeMax: "Score", curMax: "Score"):  # [board.id, stack.id, score]
        """第一个参数为下一个元素, 第二个参数为当前最大, 返回1 表示替换最大, 返回-1表示不替换
        """
        if mayBeMax.score == mayBeMax.score:#abs(mayBeMax.score - mayBeMax.score) < Constants.minimum10E_5:  # 当分数相差不大
            if (mayBeMax.board and curMax.board) is not None:  # 当板材都存在
                if (mayBeMax.stack and curMax.stack) is not None:  # 当栈都存在
                    # 考察他们的贴合情况, 能在这比较的, 都是可以push的
                    mItemW = mayBeMax.item.minLength if not mayBeMax.rotate else mayBeMax.item.maxLength
                    mStackW = mayBeMax.stack.getTopAvailableBound()[0] if mayBeMax.stackTopOrRight == Constants.top else mayBeMax.stack.getRigthAvailableBound()[0]

                    cItemW = curMax.item.minLength if not curMax.rotate else curMax.item.maxLength
                    cStackW = curMax.stack.getTopAvailableBound()[0] if curMax.stackTopOrRight == Constants.top else curMax.stack.getRigthAvailableBound()[0]
                    # 当两个栈的堆叠比率差不多,优先堆右侧的
                    if abs(mItemW / mStackW - cItemW / cStackW) < Constants.minimum10E_4:
                        if mayBeMax.stackTopOrRight == Constants.right and curMax.stackTopOrRight == Constants.top:
                            return 1
                        elif mayBeMax.stackTopOrRight == Constants.top and curMax.stackTopOrRight == Constants.right:
                            return -1
                        # 当两者都堆右侧, 优先堆宽度长的, 若宽度都相等, 则先堆旧的
                        else:
                            if mStackW != cStackW:
                                return mStackW - cStackW
                            else:
                                return -(mayBeMax.board.id - curMax.board.id)
                    else:  # 当堆叠比率有差别, 返回堆叠比率较大的
                        return mItemW / mStackW - cItemW / cStackW

                else:  # 当至少有一个栈不存在, 优先使用有栈的
                    # print("当至少有一个栈不存在, 优先使用有栈的")
                    if not mayBeMax.stack and curMax.stack:
                        return 1
                    elif mayBeMax.stack and not curMax.stack:
                        return -1
                    else:  # 当两个栈都不存在,则必然要新建栈,则选择剩余空间最小者新建
                        # 当他们的剩余空间比都差不多时,选择不翻转的那个
                        if abs(mayBeMax.item.getArea() / mayBeMax.board.remainArea() - curMax.item.getArea() / curMax.board.remainArea()) < Constants.minimum10E_4:
                            if not mayBeMax.rotate:  # 下个分数不翻转, 替换
                                return 1
                            else:
                                return -1  # 要翻转,不替换
                        else:

                            return mayBeMax.item.getArea() / mayBeMax.board.remainArea() - curMax.item.getArea() / curMax.board.remainArea()
            else:  # 前两个if让空board始终排在最后
                if not mayBeMax.board and curMax.board:  # 下块板材不存在, 不替换
                    return -1
                elif mayBeMax.board and not curMax.board:  # 下块板材存在 但当前板材不存在, 替换
                    return 1
                else:  # 若两块板材都不存在, 则判断是否需要翻转, 这是唯一的
                    if not mayBeMax.rotate:  # 下个分数不翻转, 替换
                        return 1
                    else:
                        return -1  # 要翻转,不替换
        else:
            return mayBeMax.score - curMax.score


    @staticmethod
    def forItemScore3(mayBeMax: "Score", curMax: "Score"):# [board.id, stack.id, score]

        if mayBeMax.score == mayBeMax.score:

            if mayBeMax.board is None:
                return -1
            if curMax.board is None:
                return 1
            # print(f"maybe={mayBeMax.board.id}")
            # print(f"current={curMax.board.id}")
            if mayBeMax.board.id == curMax.board.id:
                # print("ok")
                if mayBeMax.stack is None:
                    return -1
                if curMax.stack is None:
                    return 1
                return curMax.stack.id - mayBeMax.stack.id
            return curMax.board.id-mayBeMax.board.id
        else:
            return mayBeMax.score - curMax.score

    @staticmethod
    def forItemScore(mayBeMax: "Score", curMax: "Score"):# [board.id, stack.id, score]
        """第一个参数为下一个元素, 第二个参数为当前最大, 返回1 表示替换最大, 返回-1表示不替换"""
        return mayBeMax.score - curMax.score

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
    def __init__(self, W, H, identity, items: "list[Stack]" = None):
        self.stacks: "list[Stack]" = items if items is not None else []  # [ [[x, y, width, height,id],[x, y, width, height,id]]]
        self.width = W
        self.height = H
        self.id = identity  # uuid.uuid1().__str__()[:8]

    # def isLegal(self,posi,item:"Item"):

    def createStack(self, item: "Item"):
        item.atBoard = self
        self.stacks.append(Stack(item, x=self.stacks[-1].getBaseRight() if len(self.stacks) > 0 else 0))

    def __repr__(self):

        return f"{self.stacks}"

    def canAddNewStack(self, item: "Item"):
        if len(self.stacks) == 0:
            return item.minLength <= self.width
        else:
            return self.stacks[-1].getBaseRight() + item.minLength <= self.width

    def remainArea(self):
        return self.width * self.height - sum(stack.getArea() for stack in self.stacks)

    # def getTable(self):  # 获取 表格
    #
    #     li = []
    #     for stack in self.stacks:
    #         li += stack.getTableQ1()
    #     return li


class Statistics:
    def __init__(self, data: "np.ndarray",Q=1):
        self.Q=Q
        self.boards: "list[Board]" = []
        self.items: "list[Item]" = [Item(data[i],Q=Q) for i in range(len(data))]

    def calcUseRate(self):
        return sum(b.remainArea() for b in self.boards) / (H * W * len(self.boards))

    def remainArea(self):
        return sum([b.remainArea() for b in self.boards])
        pass

    def addNewBoard(self):
        board = Board(W, H, len(self.boards))
        self.boards.append(board)
        return board

    def getTable(self):
        table = []
        for board in self.boards:
            for stack in board.stacks:
                table+=stack.getTable(self.Q)
        return table
class Stack:
    """这是一个二维栈,栈顶会有两个,上侧一个,右侧已鞥"""

    def __init__(self, data: "Item", x=0):
        self.data: "list[list[BoardItem]]" = []
        self.setBase(x, data)
        self.push = {
                0: self.pushToTop,
                1: self.pushToRight
        }
        self.id = len(data.atBoard.stacks)

    def getArea(self):
        s = 0
        for row in self.data:
            for item in row:
                s += item.w * item.h
        return s

    def getTable(self,Q=1):
        t = []

        if Q == 1:  # 获取第一问的表  原片材质,原片序号,产品id,产品x坐标,产品y坐标,产品x方向长度,产品y方向长度
            rowGen = lambda i:[i.item.material_id, i.item.atBoard.id, i.id, i.x, i.y, i.w, i.h]
        elif Q ==2: # 批次序号,原片材质,原片序号,产品id,产品x坐标,产品y坐标,产品x方向长度,产品y方向长度
            rowGen = lambda b:[b.item.seires_id,b.item.material_id,b.item.atBoard.id,b.item.id,b.x,b.y,b.w,b.h]
        else:
            raise ValueError("未知的问题类型")
        for row in self.data:
            for i in row:
                    t.append(rowGen(i))
        return t

    def getBaseRight(self):
        return self.data[0][0].w + self.data[0][0].x

    def setBase(self, x, item: "Item"):
        if len(self.data) == 0:
            self.data.append([BoardItem(x, 0, item.minLength, item.maxLength, item.id, item)])
        else:
            self.data[0] = [BoardItem(x, 0, item.minLength, item.maxLength, item.id, item)]

    def getTopAvailableBound(self):
        if len(self.data) == 1:
            base = self.data[0][0]
            return [base.w, H - base.h]
        else:
            # top
            topItem = self.data[-1][0]
            topAvailaleHeight = H - (topItem.y + topItem.h)
            topAvailaleWidth = sum(item.w for item in self.data[-1])
            return [topAvailaleWidth, topAvailaleHeight]

    def getRigthAvailableBound(self):
        if len(self.data) == 1:
            return [-1, -1]
        else:
            rightItem = self.data[-1][-1]
            rightAvailaleHeight = rightItem.h
            rightAvailaleWidth = sum(item.w for item in self.data[-2]) - sum(item.w for item in self.data[-1])
            return [rightAvailaleWidth, rightAvailaleHeight]

    def pushToRight(self, item: "Item"):
        x = self.data[-1][-1].x + self.data[-1][-1].w
        y = self.data[-1][-1].y
        self.data[-1].append(BoardItem(x, y, item.minLength, item.maxLength, item.id, item))
        item.atBoard = self.data[0][0].item.atBoard

    def pushToTop(self, item: "Item"):
        x = self.data[-1][0].x
        y = self.data[-1][0].y + self.data[-1][0].h
        self.data.append([BoardItem(x, y, item.minLength, item.maxLength, item.id, item)])
        item.atBoard = self.data[0][0].item.atBoard

    def topCanPush(self, item: "Item"):
        bounds = self.getTopAvailableBound()
        return item.minLength <= bounds[0] and item.maxLength <= bounds[1]

    def rightCanPush(self, item: "Item"):
        bounds = self.getRigthAvailableBound()
        return item.minLength <= bounds[0] and item.maxLength == bounds[1]

    def __repr__(self):
        return f"stack:{[[[i.x, i.y, i.w, i.h] for i in row] for row in self.data]}"

    def __str__(self):
        return self.__repr__()


@dataclasses.dataclass
class Score:
    item: "Item"
    # stackScore: float
    # boardScore: float
    score: float
    board: Board = None  # 如果为负, 则表示新建board
    stack: Stack = None  # 如果为空, 则表示新建stack
    rotate: bool = False
    stackTopOrRight: int = Constants.top
    stackWeight: "Union[int,float]" = 2
    boardWeight: "Union[int,float]" = 1

    # @property
    # def score(self):
    #     return self.stackScore * self.stackWeight + self.boardScore * self.boardWeight

class Run:

    @staticmethod
    def valueFunc(item: "Item", board: "Board" = None, bound: "list" = None, topOrRight=Constants.top, stackWeight=2, boardWeight=1) -> "float":
        """
        设置一个价值函数,计算一个零件在一个板材中的价值, 每次都选择最优的放置点.
        优先stack, 其次
        :param item:
        :param board:
        :return:
        """
        byStackWidthRatio = (item.minLength / bound[0]) if bound is not None else 0
        byStackRemainAreaRatio =(item.getArea()/(bound[0]*bound[1])) if bound is not None else 0
        byBoardRemainAreaRatio = (item.getArea() / board.remainArea()) if board is not None else 0
        boardRemainWidth = board.width - board.stacks[-1].getBaseRight() if board is not None else 0
        byBoardRemainWidthRatio = (item.minLength / boardRemainWidth) if board is not None and boardRemainWidth > 0 else 0
        #
        # stackScore = (2 if topOrRight == Constants.right else 1)*stackWeight * (byStackWidthRatio+byStackRemainAreaRatio)
        # boardScore = boardWeight * (byBoardRemainAreaRatio+byBoardRemainWidthRatio)
        # 下面这套方案 D组最高利用率达到79.3% 2560块
        stackScore = (2 if topOrRight == Constants.right else 1) * stackWeight * (byStackWidthRatio+ byStackRemainAreaRatio )
        boardScore = boardWeight * (byBoardRemainAreaRatio)


        return stackScore + boardScore

    @staticmethod
    def valueFunc2(item: "Item", board: "Board" = None, stack: "list[BoardItem]" = None, stackWeight=2, boardWeight=1) -> "float":
        """
        设置一个价值函数,计算一个零件在一个板材中的价值, 每次都选择最优的放置点.
        优先stack, 其次
        :param item:
        :param board:
        :return:
        """
        stackScore = ((stackWeight * item.minLength / stack[-1].w) if stack is not None else 0)
        boardScore = (boardWeight * (item.minLength * item.maxLength) / board.remainArea()) if stack is not None else boardWeight * (item.minLength * item.maxLength) / (H * W)
        return stackScore + boardScore

    @staticmethod
    def addScore(item, scoreLi, boards, rightPush=True, rotate=False):
        for board in boards:
            for stack in board.stacks:
                if stack.rightCanPush(item):
                    scoreLi.append(Score(item, Run.valueFunc(item, board=board, bound=stack.getRigthAvailableBound(),topOrRight=Constants.right),
                                         board=board, stack=stack, stackTopOrRight=Constants.right, rotate=rotate))
                if stack.topCanPush(item):
                    scoreLi.append(Score(item, Run.valueFunc(item, board=board, bound=stack.getTopAvailableBound()),
                                         board=board, stack=stack, rotate=rotate))
            if board.canAddNewStack(item):
                scoreLi.append(Score(item, Run.valueFunc(item, board=board), board=board, rotate=rotate))
    @staticmethod
    def setItemToboard(higherScore:"Score",stats:"Statistics"):
        item = higherScore.item
        if higherScore.rotate: item.switchLength()  # 如果需要转置矩形
        if higherScore.board is None:
            stats.addNewBoard().createStack(item)
        elif higherScore.stack is None:
            higherScore.board.createStack(item)
        else:
            if higherScore.stackTopOrRight == Constants.top:
                higherScore.stack.pushToTop(item)
            else:
                higherScore.stack.pushToRight(item)

    # @staticmethod
    # def v2(stats: "Statistics", cmpFun=Cmp.forItemScore3):
    #     # 本代码是提交论文时的代码
    #     """加入评分机制来选择板块
    #
    #         items 为n行2列矩阵
    #         """
    #     itemsli = stats.items
    #     itemsli.sort(key=lambda x: x[0].minLength)
    #     valueFunc = Run.valueFunc
    #     for i in range(len(itemsli) - 1, -1, -1):
    #         item = itemsli[i][0]
    #         if item.atBoard is not None: continue
    #         # 开始评分
    #         scoreLi = [[-1, -1, valueFunc(item), False]]
    #         for board in stats.boards:
    #             for s in range(len(board.stacks)):  # 如果有stack 则对不同的stack放置打分
    #                 if board.canStack(item, board.stacks[s]):
    #                     scoreLi.append([board.id, s, valueFunc(item, board, board.stacks[s]), False])
    #             if board.canAddNewStack(item):  # 如果可以作为新stack的base, 则打分
    #                 scoreLi.append([board.id, -1, valueFunc(item, board), False])
    #         item.switchLength()
    #         scoreLi += [[-1, -1, valueFunc(item), True]]  # TODO 交换长宽需要作为排序依据
    #         for board in stats.boards:
    #             for s in range(len(board.stacks)):  # 如果有stack 则对不同的stack放置打分
    #                 if board.canStack(item, board.stacks[s]):
    #                     scoreLi.append([board.id, s, valueFunc(item, board, board.stacks[s]), True])
    #             if board.canAddNewStack(item):  # 如果可以作为新stack的base, 则打分
    #                 scoreLi.append([board.id, -1, valueFunc(item, board), True])
    #         item.switchLength()
    #
    #         # 开始选取最高分
    #         higherScore = max(scoreLi, key=cmp_to_key(Cmp.forItemScore))
    #
    #         if higherScore[3]: item.switchLength()  # 如果需要转置矩形
    #         if higherScore[0] == -1:
    #             board = stats.addNewBoard()
    #             board.createStack(item)
    #         elif higherScore[1] == -1:
    #             board = stats.boards[higherScore[0]]
    #             board.createStack(item)
    #         else:
    #             board = stats.boards[higherScore[0]]
    #             board.addItemToStack(item, board.stacks[higherScore[1]])
    #
    #     return stats

    @staticmethod
    def v3(stats: "Statistics",cmpFun=Cmp.forItemScore):
        """加入评分机制来选择板块

        items 为n行2列矩阵
        """
        itemsli = stats.items
        itemsli.sort(key=cmp_to_key(Cmp.forItemSort))

        for i in range(len(itemsli)):
            item = itemsli[i]
            # 开始评分
            allScore = []
            allScore+=[Score(item, Run.valueFunc(item)), Score(item, Run.valueFunc(item), rotate=True)]
            Run.addScore(item, allScore, stats.boards)

            item.switchLength()
            Run.addScore(item, allScore, stats.boards, rotate=True)
            item.switchLength()

            # 开始选取最高分
            higherScore: "Score" = max(allScore, key=cmp_to_key(cmpFun))

            # 开始插入
            Run.setItemToboard(higherScore,stats)
            # Utils.print(f"已跑第{i}轮")

        return stats

    @staticmethod
    def v4(stats: "Statistics",cmpFun=Cmp.forItemScore):
        """先对每个配件在每个可能位置进行打分,排样延迟到循环结束后, """
        itemsli = stats.items
        itemsli.sort(key=lambda x: x.minLength, reverse=True)
        # valueFunc = Utils.valueFunc
        keepChange = True
        while len(itemsli) > 0:
            # keepChange=False
            allScore = []
            for i in range(len(itemsli)):
                item = itemsli[i]

                allScore += [Score(item, Run.valueFunc(item)), Score(item, Run.valueFunc(item), rotate=True)]
                Run.addScore(item, allScore, stats.boards)

                item.switchLength()
                Run.addScore(item, allScore, stats.boards, rotate=True)
                item.switchLength()

            # 开始选取最高分
            higherScore: "Score" = max(allScore, key=cmp_to_key(cmpFun))

            # 开始插入
            Run.setItemToboard(higherScore,stats)
            itemsli = [item for item in itemsli if item.atBoard is None]
            Utils.print(f"{item}确定,剩余{len(itemsli)}")
        return stats

def Q(q=1,row=2000,col=6):
    """随机生成配件排样"""
    if q==Constants.random:


        # 长尾分布样本
        maxLengths = Utils.Zipf(1,20,H,row)
        minLengths = Utils.Zipf(1,20,W,row)

        # 正态分布样本
        # Hmeans,Hvar,Wmeans,Wvar=H/2+20,550,W/2+20,250
        # maxLengths = norm.ppf(np.random.random(row*3), loc=Hmeans, scale=Hvar).astype(int)
        # maxLengths = maxLengths[maxLengths<=H]
        # maxLengths = maxLengths[20 <=maxLengths][:row]
        # minLengths = norm.ppf(np.random.random(row*3), loc=Wmeans, scale=Wvar).astype(int)
        # minLengths = minLengths[minLengths <= W]
        # minLengths = minLengths[20 <= minLengths][:row]
        # ax = plt.gca()
        # ax.get_xaxis().get_major_formatter().set_scientific(False)
        # plt.hist(minLengths,bins=1000)
        # plt.show()
        # print(maxLengths)

        # # 均匀分布样本
        # maxLengths = np.random.randint(1,H,size=row)
        # minLengths = np.random.randint(1,W,size=row)

        items = np.ones((row,col-2))
        items = np.insert(items,3, maxLengths, axis=1)
        items = np.insert(items,3, minLengths, axis=1)
        stats = Statistics(items)
        Run.v4(stats)
        ratio = 1 - stats.remainArea() / (len(stats.boards) * (H * W))
        sum = len(stats.boards)
        Utils.print(ratio,sum)
        # Utils.log([ratio,sum,f"正态均值:{Hmeans},{Wmeans}",f"方差:{Hvar},{Wvar}"])
        Utils.drawStats(stats,"zipf")

def Q1():
    for i in range(1, 6):
        items = Utils.readCsv(datapathA(i))
        stats = Statistics(items)
        Run.v3(stats)
        # Utils.saveBoardAsCsv(stats, datapathA(i), items)
        Utils.print(1 - stats.remainArea() / (len(stats.boards) * (H * W)))
        Utils.print(len(stats.boards))

def Q2(run=Run.v3):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    ax = plt.gca()
    for i in range(1,6):
        items = Utils.readCsv(datapathB(i)) # [series_id,item_order,item_material,item_id,max_length,min_length, series_id]
        avg = 0
        allBoardsCount = 0
        allStats= []
        allTable = []
        totalBoardArea=0
        totalFreeArea=0
        seriesNames = set(items[:,0])
        # print(seriesNames)
        Utils.print(f"开始执行第B{i}个数据集的排样")
        for seriesName in seriesNames:
            sameSeriesItems =np.array([i for i in items if i[0]==seriesName]) #items[[i[0]==seriesName for i in items]]
            materialNames = set(sameSeriesItems[:,2])
            for materialName in materialNames:
                sameMaterialItems =np.array([i for i in sameSeriesItems if i[2]== materialName])#items[[i[1]==materialName for i in sameSeriesItems]]
                stats = Statistics(sameMaterialItems,Q=2)
                allStats.append(run(stats))
        for oneStats in allStats:
            allTable+=oneStats.getTable()
            totalBoardArea+=len(oneStats.boards)*(H*W)
            totalFreeArea+=oneStats.remainArea()
            allBoardsCount += len(oneStats.boards)
        Utils.print(f"排样完成,本批次利用率:{1 - totalFreeArea / totalBoardArea},共计使用板材={allBoardsCount}")
        # Constants.imgCapacity = 0
        # fig = plt.figure(figsize=(34, 12))
        # for s in allStats:
        #     for b in s.boards:
        #         plt.subplot(Constants.imgRow, Constants.imgCol, (Constants.imgCapacity % (Constants.imgRow * Constants.imgCol)) + 1)
        #         Utils.drawBoard(b)
        #         Constants.imgCapacity += 1
        #         if Constants.imgCapacity % (Constants.imgRow * Constants.imgCol) == 0:
        #             plt.savefig(f"dataB{i}_{int(Constants.imgCapacity / (Constants.imgRow * Constants.imgCol))}.png",bbox_inches='tight')
        #             plt.clf()
        # if Constants.imgCapacity % (Constants.imgRow * Constants.imgCol) != 0:
        #     plt.savefig(f"dataB{i}_{int(Constants.imgCapacity/(Constants.imgRow*Constants.imgCol))+1}.png",bbox_inches='tight')
        #     plt.clf()
def Q3(run=Run.v3):
    for i in range(1,2):
        items = Utils.readCsv(datapathB(i))  # [series_id,item_order,item_material,item_id,max_length,min_length, series_id]
    # plt.rcParams["font.sans-serif"] = ["SimHei"]
    # ax = plt.gca()
    # for i in range(1, 6):
    #     items = Utils.readCsv(datapathB(i))  # [series_id,item_order,item_material,item_id,max_length,min_length, series_id]
        avg = 0
        allBoardsCount = 0
        allStats = []
        allTable = []
        totalBoardArea = 0
        totalFreeArea = 0
        seriesNames = set(items[:, 0])
    #     # print(seriesNames)
        Utils.print(f"开始执行第B{i}个数据集的排样")
        materialNames = set(items[:, 1])
        for materialName in materialNames:
            sameMaterialItems = np.array([i for i in items if i[1] == materialName])  # items[[i[1]==materialName for i in sameSeriesItems]]
            stats = Statistics(sameMaterialItems, Q=2)
            allStats.append(run(stats))

        for oneStats in allStats:
            allTable += oneStats.getTable()
            totalBoardArea += len(oneStats.boards) * (H * W)
            totalFreeArea += oneStats.remainArea()
            allBoardsCount += len(oneStats.boards)
        Utils.print(f"排样完成,本批次利用率:{1 - totalFreeArea / totalBoardArea},共计使用板材={allBoardsCount}")
        writer= writer = csv.writer(open(f"直接统计.csv", 'a', encoding='UTF8', newline=''))
        writer.writerow([f"{datetime.now()}",f"dataB{i}",1 - stats.remainArea() / (len(stats.boards) * (H * W)),len(stats.boards)])

if __name__ == "__main__":
    Utils.print("开始")
    for i in range(1):
        Q(q=Constants.random)
    # Q2()
    # Utils.sortCsv()
    # Utils.print()

        # Utils.print((stats.getTable()[:10]))

        #
        # for b in stats.boards:
        #     Utils.drawBoard(b)

    pass
