# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = 'v5第二问.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/9 11:07'
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import MultipleLocator
import numpy as np
from datetime import datetime
from functools import cmp_to_key
import csv
# from core import v2_run
W = 1220
H = 2440
datapathA = lambda at: f"./B题数据/子问题1-数据集A/dataA{at}.csv"
datapathB = lambda at: f"./B题数据/子问题2-数据集B/dataB{at}_sorted.csv"
seriesPathB = lambda at: f"./B题数据/子问题2-数据集B/dataB{at}_sorted.csv"
dataBFinalOutPutPath = lambda at: f"./B题数据/子问题2-数据集B/cut_program_dataB{at}.csv"



class Constants:
    imgCapacity = 0
    imgRow = 5
    imgCol = 20

class Utils:
    @staticmethod
    def sortCsv():
        """主要将第2题的文件进行按材质分类,"""
        datapath = lambda at: f"./B题数据/子问题2-数据集B/dataB{at}.csv"

        for i in range(1, 5):

            items = Utils.readCsv(datapath(i))
            writer = csv.writer(open(datapath(i)[:-4] + "_sorted.csv", 'w', encoding='UTF8', newline=''))
            writer.writerow(["item_id", "item_material", "max_length", "min_length", "item_order", "total"])
            maxlength = np.array([max(i[0], i[1]) for i in items[:, 3:5].astype(np.float_)])
            minlength = np.array([min(i[0], i[1]) for i in items[:, 3:5].astype(np.float_)])
            items = np.delete(items, [3, 4], axis=1)
            items = np.insert(items, 3, maxlength, axis=1)
            items = np.insert(items, 4, minlength, axis=1)
            items = sorted(items,key=cmp_to_key(Cmp.forCsvSort))
            # print(items[30:90])
            writer.writerows(items)
            # orders = set(items[:,5])
            # orderItems = {}
            # for order in orders:
            #     ouputOrderItem = items[[i[5]==order for i in items]]
            #     ouputitems = {}
            #     materials = set(ouputOrderItem[:, 1])
            #     for material in materials:
            #         ouputMaterialItem =items[[i[1]==material for i in ouputOrderItem]]
            #         item_order = ouputMaterialItem[:, 5]
            #         maxlength = np.array([max(i[0],i[1]) for i in ouputMaterialItem[:,3:5].astype(np.float_)])
            #         minlength = np.array([min(i[0],i[1]) for i in ouputMaterialItem[:,3:5].astype(np.float_)])
            #         ouputMaterialItem = np.delete(ouputMaterialItem, [2,3,4,5], axis = 1)
            #         ouputMaterialItem = np.insert(ouputMaterialItem, 2, maxlength,  axis=1)
            #         ouputMaterialItem = np.insert(ouputMaterialItem, 3, minlength,  axis=1)
            #         ouputMaterialItem = np.insert(ouputMaterialItem, 4, item_order, axis=1)
            #         total = np.size(ouputMaterialItem[:,0])
            #         ouputMaterialItem = np.insert(ouputMaterialItem,5,(np.ones(total)*total).astype(np.int32),axis=1)
            #         index = np.lexsort((ouputMaterialItem[:,3].astype(np.float_),))
            #         ouputMaterialItem = ouputMaterialItem[index]
            #         print(ouputMaterialItem)
                    # writer.writerows(ouputMaterialItem)
        #

            # for i in ouputitems.keys():
            #     writer.writerows(ouputitems[i])
            # print(set(items[:,1]))
            # items = sorted(items,key=lambda x:x[1])
            # print(items[])



    @staticmethod
    def print(*args, **kwargs):
        print(f"{datetime.now()}")
        print(*args, **kwargs)

    @staticmethod
    def satisfiesCondition(items: "np.ndarray"):
        # print(items)
        totalArea = 250 * 1000 * 1000
        totalNum = 1000
        # print(np.dot(items[:,0], items[:,1]) )
        return np.dot(items[:, 0], items[:, 1]) <= totalArea and len(items) <= totalNum


    @staticmethod
    def readCsv(path):
        return np.loadtxt(path, dtype=np.str_, delimiter=',', skiprows=1, encoding='utf-8')

    @staticmethod
    def saveBoardAsCsv(stats: "Statistics", csvFrom: "str", items: "np.ndarray"):
        title = ["原片材质", "原片序号", "产品id", "产品x坐标", "产品y坐标", "产品x方向长度", "产品y方向长度"]
        data = []

        for b in stats.boards:
            data += b.get2DList(items)

        writer = csv.writer(open(f"{csvFrom[:-4]}_cut_program.csv", 'w', encoding='UTF8', newline=''))
        writer.writerow(title)
        writer.writerows(data)
        pass

    # @staticmethod
    # def drawBoard(b: "Board"):
    #     fig = plt.figure(figsize=(8, 6))
    #
    #     ax = plt.gca()
    #     # plt.pause(0.1)
    #     # fig, ax = plt.subplots()
    #     # ax.plot([], [])
    #     x_major_locator = MultipleLocator(100)
    #     # 把x轴的刻度间隔设置为1，并存在变量里
    #     y_major_locator = MultipleLocator(200)
    #     ax.xaxis.set_major_locator(x_major_locator)
    #     # 把x轴的主刻度设置为1的倍数
    #     ax.yaxis.set_major_locator(y_major_locator)
    #
    #     plt.xlim(0, b.width)
    #     plt.ylim(0, b.height)
    #     for boardItemLi in b.stacks:
    #         for boardItem in boardItemLi:
    #             ax.add_patch(patches.Rectangle(
    #                     (boardItem.x, boardItem.y),
    #                     boardItem.w,
    #                     boardItem.h,
    #                     edgecolor='blue',
    #                     facecolor='white',
    #                     fill=True
    #             ))
    #
    #     fig.savefig(f"{datetime.now()}.png".replace(":", "-"))
    @staticmethod
    def drawBoard(b: "Board", word="", saveName=""):
        # 4*10, word= series:10,material:Abc,board_id:
        # if Constants.imgCapacity == 0:
        #     fig = plt.figure(figsize=(24, 12))
        # plt.subplot(Constants.imgRow,Constants.imgCol,(Constants.imgCapacity % (Constants.imgRow*Constants.imgCol))+1)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.xlim(0, W)
        plt.ylim(0, H)

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
        m = b.stacks[0][0].item.material_id
        s = b.stacks[0][0].item.seires_id

        plt.title(f"{s}_{m}_{b.id}")
        # Constants.imgCapacity+=1
        # if Constants.imgCapacity % (Constants.imgRow*Constants.imgCol) == 0:
        #     plt.savefig(f"{saveName}_{int(Constants.imgCapacity/(Constants.imgRow*Constants.imgCol))}.png")
        #     plt.clf()
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
    def forCsvSort(_next,_current):# [0item_id,1material,2item_num,3max_length,4min_length,5item_order]
        """返回1表示替换,返回-1表示不替换,order->material->min_length"""
        if _next[5]>_current[5]: return 1
        elif _next[5]<_current[5]: return -1
        else:
            if _next[1] > _current[1]:
                return 1
            elif _next[1] < _current[1]:
                return -1
            else:
                return float( _next[4]) - float(_current[4])
                # if _next[4] > _current[4]:
                #     return 1
                # elif _next[4] < _current[4]:
                #     return -1

    @staticmethod
    def forQ2Sort(_next,_current):# [series, material, board_id , item_id,]
        """返回1表示替换,返回-1表示不替换,order->material->min_length"""
        if _next[5] > _current[5]:
            return 1
        elif _next[5] < _current[5]:
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
    def __init__(self, seires_id, item_order,item_material,item_id,maxLength,minLength):
        self.maxLength = float(maxLength)
        self.minLength = float(minLength)  # 作为条带的基底
        self.id = item_id  # 如果是组合,则用最低的
        self.order_id = item_order
        self.material_id = item_material
        self.seires_id = seires_id
        self.atBoard = None

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
        self.item:"Item" = item
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
        self.stacks.append([BoardItem(self.atLastStackRight(), 0, item.minLength, item.maxLength, item.id,item)])
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

    def getTable(self):
        table=[]
        for stack in self.stacks:
            table+=[[b.item.seires_id,b.item.material_id,self.id,b.item.id,b.x,b.y,b.w,b.h] for b in stack]

        return table





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

    def get2DList(self, rawData: "np.ndarray"):

        li = []
        for stack in self.stacks:
            # 材质,序号,订单编号,x,y,w,h

            for item in stack:
                # item_id = rawData[item.id, 0]
                materialName = rawData[item.id, 1]
                li.append([materialName, self.id, item.id, item.x, item.y, item.w, item.h])
        return li


class Statistics:
    def __init__(self, data:"np.ndarray"):# [series_id,item_order,item_material,item_id,max_length,min_length, series_id]
        self.boards: "list[Board]" = []
        self.items: "list[list[Item]]" = [[Item(*data[i])] for i in range(len(data))]
        self.originalData = data

    def calcUseRate(self):
        return sum(b.freeArea() for b in self.boards) / (H * W * len(self.boards))

    def freeArea(self):
        return sum([b.freeArea() for b in self.boards])
        pass

    def addNewBoard(self):
        board = Board(W, H, len(self.boards))
        self.boards.append(board)
        return board

    def getTable(self):
        table= []
        for board in self.boards:
            table+=board.getTable()
        return table



def v2_run(stats:"Statistics"):
    """加入评分机制来选择板块

    items 为n行2列矩阵
    """
    allPlate = []
    # stats = Statistics(items)
    itemsli = stats.items
    itemsli.sort(key=lambda x: x[0].minLength)
    valueFunc = Utils.valueFunc
    for i in range(len(itemsli) - 1, -1, -1):
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

# from core import v2_run
# from v2_core import Run,Statistics,Cmp,Utils

if __name__ == "__main__":
    Utils.print()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    ax = plt.gca()
    # plt.tick_params(bottom=False, top=False, left=False, right=False)
    # ax.tick_params(bottom=False, top=False, left=False, right=False)



    for i in range(1,2):
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
            print(f"序列:{seriesName},{materialNames}")
            for materialName in materialNames:
                sameMaterialItems =np.array([i for i in sameSeriesItems if i[2]== materialName])#items[[i[1]==materialName for i in sameSeriesItems]]
                stats = Statistics(sameMaterialItems)
                allStats.append(v2_run(stats))
        for oneStats in allStats:
            allTable+=oneStats.getTable()
            totalBoardArea+=len(oneStats.boards)*(H*W)
            totalFreeArea+=oneStats.freeArea()
            allBoardsCount += len(oneStats.boards)

        Utils.print(f"排样完成,本批次利用率:{1-totalFreeArea/totalBoardArea},共计使用板材={allBoardsCount},下面将排样数据写入csv文件")

        writer = csv.writer(open(dataBFinalOutPutPath(i), 'w', encoding='UTF8', newline=''))
        writer.writerow(["批次序号","原片材质","原片序号","产品id","产品x坐标","产品y坐标","产品x方向长度","产品y方向长度"])
        writer.writerows(allTable)
        # for b in allStats[0].boards:
        #     Utils.drawBoard(b)
        # Utils.print()
        # Utils.print("下面保存排样方案图示")
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

    pass
