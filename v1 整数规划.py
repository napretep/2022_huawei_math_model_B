# -*- coding: utf-8 -*-
"""
__project_ = 'pythonscripts'
__file_name__ = 'v1 整数规划.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2022/10/6 14:01'
"""
from typing import Optional

from scipy import optimize as op
import numpy as np
import pulp as pp



var = pp.LpVariable
problem = pp.LpProblem
minimize = pp.LpMinimize
binary = pp.LpBinary
equal = pp.LpConstraintEQ
lessEqual = pp.LpConstraintLE
greaterEqual = pp.LpConstraintGE


csvDir = "./B题数据/子问题1-数据集A/dataA1.csv"

def varMaker(n):
    # α β γ δ
    realN = n+1
    _1_n = (1,realN)
    varDict={
            "α":{},
            "β":{},
            "γ":{},
            "δ":{},
    }
    for k in range(1,n+1): # beta
        for j in range(k,n+1):
            betaName=f"β_{k}_{j}"
            varDict["β"][betaName]=var(betaName,cat=binary)
    for j in range(1,n+1): # alpha
        for i in range(j,n+1):
            if i>=j:
                alphaName=f"α_{j}_{i}"
                varDict["α"][alphaName] = var(alphaName, cat=binary)
    for l in range(*_1_n): # gamma
        for k in range(l,n+1):
            gammaName = f"γ_{l}_{k}"
            varDict["γ"][gammaName] = var(gammaName, cat=binary)

    for l in range(1,n):
        for i in range(l+1,n+1):
            for j in range(l,i):
                if l == n:
                    continue
                if i >= l+1:
                    deltaName= f"δ_{l}_{i}_{j}"
                    varDict["δ"][deltaName] = var(deltaName, cat=binary)
    return varDict

def goalMaker(n,minProblem:"pp.LpProblem",varDict):
    realN = n + 1
    _1_n = (1, realN)
    for i in range(*_1_n):
        minProblem+=varDict["γ"][f"γ_{i}_{i}"]

def bound_17(n,minProblem:"pp.LpProblem",varDict):
    realN = n + 1
    _1_n = (1, realN)
    for i in range(*_1_n):
        varSum:"Optional[var]" = None
        for j in range(*_1_n):
            if j<=i:
                alpha_j_i = varDict["α"][f"α_{j}_{i}"]
                if varSum is None:
                    varSum=alpha_j_i
                else:
                    varSum+=alpha_j_i
        minProblem += varSum==1

def bound_18(n,minProblem:"pp.LpProblem",varDict):
    for j in range(1,n):
        varSum: "Optional[var]" = None
        for i in range(j+1,n+1):
            alpha_j_i = varDict["α"][f"α_{j}_{i}"]
            if varSum is None:
                varSum = alpha_j_i
            else:
                varSum += alpha_j_i
        alpha_j_j = varDict["α"][f"α_{j}_{j}"]
        minProblem += varSum <=(n-j)*alpha_j_j

    pass
def bound_19(n,minProblem:"pp.LpProblem",varDict, w:"list",h:"list",H):

    for j in range(1,n):
        for i in range(j+1,n+1):
            if w[i]!=w[j] or h[i]+h[j]>H:
                alpha_j_i = varDict["α"][f"α_{j}_{i}"]
                minProblem += alpha_j_i==0
    pass


def bound_20(n,minProblem:"pp.LpProblem",varDict):
    for j in range(1,n+1):
        varSum: "Optional[var]" = None
        for k in range(1,n+1):
            beta_k_j = varDict["β"][f"β_{k}_{j}"]
            if varSum is None:
                varSum = beta_k_j
            else:
                varSum += beta_k_j
        alpha_j_j = varDict["α"][f"α_{j}_{j}"]
        minProblem += varSum <= (n - j) * alpha_j_j
    pass

def bound_21(n,minProblem:"pp.LpProblem",varDict,h:"list",H):

    for k in range(2,n+1):
        for j in range(1,k):
            leftsum: "Optional[var]" = None
            for i in range(j,n+1):
                alpha_j_i = varDict["α"][f"α_{j}_{i}"]
                if leftsum is not None:
                    leftsum+=h[i]*alpha_j_i
                else:
                    leftsum=h[i]*alpha_j_i
                pass
            rightSum: "Optional[var]" =None
            for i in range(k,n+1):
                alpha_k_i = varDict["α"][f"α_{k}_{i}"]
                rightSum = rightSum + h[i] * alpha_k_i if rightSum is not None else  h[i] * alpha_k_i

                pass
            beta_k_j = varDict["β"][f"β_{k}_{j}"]
            rightSum += (H+1)*(1-beta_k_j)
            minProblem+= leftsum<=rightSum

    pass

def bound_22(n,minProblem:"pp.LpProblem",varDict,h:"list",H):
    for k in range(1,n):
        for j in range(k+1,n+1):
            leftsum: "Optional[var]" = None
            for i in range(j, n + 1):
                alpha_j_i = varDict["α"][f"α_{j}_{i}"]
                leftsum = leftsum + h[i] * alpha_j_i if leftsum else  h[i] * alpha_j_i
            rightSum: "Optional[var]" = None
            for i in range(k,n+1):
                alpha_k_i = varDict["α"][f"α_{k}_{i}"]
                rightSum = rightSum + h[i] * alpha_k_i if rightSum else h[i] * alpha_k_i
            beta_k_j:var = varDict["β"][f"β_{k}_{j}"]
            rightSum:"var" =rightSum + H * (1 - beta_k_j)
            minProblem += leftsum <= rightSum

    pass


def bound_23(n,minProblem:"pp.LpProblem",varDict,w:"list",W):
    for k in range(1,n+1):
        leftsum: "Optional[var]" = None
        for j in range(1,n+1):
            beta_k_j = varDict["β"][f"β_{k}_{j}"]
            leftsum = leftsum+ w[j]*beta_k_j if leftsum is not None else  w[j]*beta_k_j
        beta_k_k = varDict["β"][f"β_{k}_{k}"]
        minProblem += leftsum<= W * beta_k_k
    pass

def bound_24(n,minProblem:"pp.LpProblem",varDict):
    for k in range(1,n+1):
        leftsum: "Optional[var]" = None
        for j in range(1, n + 1):
            gamma_l_k = varDict["β"][f"β_{k}_{j}"]
            leftsum = leftsum + gamma_l_k if leftsum is not None else gamma_l_k
        beta_k_k = varDict["β"][f"β_{k}_{k}"]
        minProblem += leftsum == beta_k_k


    pass

def bound_25(n,minProblem:"pp.LpProblem",varDict,h:"list",H):
    for l in range(1,n):
        leftsum: "Optional[var]" = None
        for i in range(l,n+1):
            gamma_l_i = varDict["γ"][f"γ_{l}_{i}"]
            leftsum = leftsum + h[i]* gamma_l_i if leftsum else h[i]* gamma_l_i
        for i in range(l+1,n+1):
            leftsum2: "Optional[var]" = None
            for j in range(l,i):
                delta_l_i_j = varDict["δ"][f"δ_{l}_{i}_{j}"]
                leftsum2 = leftsum2+delta_l_i_j if leftsum2 is not None else delta_l_i_j
            leftsum+=h[i]*leftsum2
        gamma_l_l = varDict["γ"][f"γ_{l}_{l}"]
        minProblem+=leftsum<=H*gamma_l_l
    pass

def bound_26(n,minProblem:"pp.LpProblem",varDict):
    for l in range(1,n):
        for i in range(l+1,n+1):
            for j in range(l,i):
                alpha_j_i = varDict["α"][f"α_{j}_{i}"]
                gamma_l_j = varDict["γ"][f"γ_{l}_{j}"]
                delta_l_i_j = varDict["δ"][f"δ_{l}_{i}_{j}"]

                minProblem+=alpha_j_i+gamma_l_j-1 <= delta_l_i_j
                minProblem+=delta_l_i_j<= (alpha_j_i+gamma_l_j)/2

    pass

def bound_27(n,minProblem:"pp.LpProblem",varDict):
    for l in range(1,n):
        leftsum = None
        for k in range(l+1,n+1):
            gamma_l_k = varDict["γ"][f"γ_{l}_{k}"]
            leftsum=leftsum+gamma_l_k if leftsum is not None else  gamma_l_k
        gamma_l_l = varDict["γ"][f"γ_{l}_{l}"]
        minProblem+= leftsum <=(n-1)*gamma_l_l

    pass


def boundMaker(n,minProblem:"pp.LpProblem",varDict,H,W,h:"list",w:"list"):
    bound_17(n, minProblem, varDict )
    bound_18(n, minProblem, varDict )
    bound_19(n, minProblem, varDict, w, h, H)
    bound_20(n, minProblem, varDict)
    bound_21(n, minProblem, varDict, h, H)
    bound_22(n, minProblem, varDict, h, H)
    bound_23(n, minProblem, varDict, w, W)
    bound_24(n, minProblem, varDict )
    bound_25(n, minProblem, varDict, h, H)
    bound_26(n, minProblem, varDict )
    bound_27(n, minProblem, varDict )

def ILPsolver(n,H,W,h:"list",w:"list"):
    minProblem = problem(name='ILP', sense=minimize)
    varDict = varMaker(n)
    goalMaker(n,minProblem,varDict)
    boundMaker(n,minProblem,varDict,H,W,h,w)

    return minProblem

if __name__ == "__main__":
    t1 = np.loadtxt(csvDir, dtype=np.float_, delimiter=',', skiprows=1, usecols=( 3, 4), encoding='utf-8')
    # csv = np.recfromcsv(csvDir,encoding="utf8",usecols=(3,4))
    # print(csv.dtype.fields)
    n = 10
    # print(t1[:10, 0:1])
    h = [i[0] for i in t1[:n, 0:1]]+[0]
    w = [i[0] for i in t1[:n, 1:2]]+[0]
    W = 1220
    H = 2440

    print(ILPsolver(n,H,W,h,w).solve())
    pass