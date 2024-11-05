import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import cos, sin, acos, asin, pi, atan, fabs, ceil, floor
from collections import defaultdict
import pulp as pl
import random
import time
from matplotlib import colors, cm
from math import ceil
import csv
import copy as cp
import cplex
from pulp import lpSum


# ROLE_NUM = 5
# ROLE_S = [0 for i in range(51)]
# VAR_S = [0 for i in range(51)]
# NUM = 1
# TEAM_NUM = 5
# TEAM_S = [[0 for x in range(10)] for y in range(25)]
# TEAMVAR_S = [[0 for x in range(10)] for y in range(25)]
# RESULT3 = [[0 for x in range(51)] for y in range(51)]


class Assignment:
    @classmethod
    # maxDrone: the amount of the drones
    def KM(cls, Q, La, L,Sl,T2023,T2024,Tjc,La2):
        row = len(Q)
        col = len(Q[0])
        print(row)
        print(col)
        # print(str(row) + "," + str(col))
        # 设置cplex求解器目标最大化
        pro = pl.LpProblem('Max(connection and coverage)', pl.LpMaximize)
        # build variables for the optimal problem
        # T矩阵设置变量
        lpvars = [[pl.LpVariable("x" + str(i) + "y" + str(j),lowBound=0,upBound=5000,
                                   cat='Integer')
                     for j in range(col)] for i in range(row)]

        binary_variables = {(i, j, k): pl.LpVariable(f"binary_{i}_{j}_{k}", cat='Binary') for i in range(row) for j in range(col) for k in [0, 2, 3, 4]}


        # build optimal function
        # 求解T矩阵
        all = pl.LpAffineExpression()
        for i in range(0, row):
            for j in range(0, col):
                all += Q[i][j] * lpvars[i][j]
        pro += all

        # 每一个任务都要被指派足够的人员
        for j in range(0, col):
                    tempSum = 0
                    for i in range(0, row):
                        tempSum += lpvars[i][j]
                    pro += tempSum == L[j]  # 这里讨论等于还是大于等于的问题

        #每个专业可以提供的最大数量
        for i in range(0, row):
            tempSum = 0
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= La[i]

        #带广东省每个专业可以提供的最大数量
        for i in range(0, row):
            tempSum = 0
            for j in range(0, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= La2[i]

        #要与2023年的指派有一定的相似性
        for i in range(0, row):
            for j in range(0,col):
                pro += lpvars[i][j] >= T2023[i][j]

        #教务处仅给出各学院配额，这里需要按照学院修改
        #机电工程学院
        tempSum = 0
        for i in range(0, 5):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 82

        #自动化学院
        tempSum = 0
        for i in range(5, 9):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 30

        #轻工化工学院
        tempSum = 0
        for i in range(9, 13):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 108

        #信息工程学院
        tempSum = 0
        for i in range(13, 14):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 16

        #土木与交通工程学院
        tempSum = 0
        for i in range(14, 20):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 104

        #计算机学院
        tempSum = 0
        for i in range(20, 21):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 38

        #材料与能源学院
        tempSum = 0
        for i in range(21, 24):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 129

        #环境科学与工程学院
        tempSum = 0
        for i in range(24, 27):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 53

        #外国语学院
        tempSum = 0
        for i in range(27, 28):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 6

        #物理与光电工程学院
        tempSum = 0
        for i in range(28, 31):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 20

        #生物医药学院
        tempSum = 0
        for i in range(31, 34):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 19

        #集成电路学院
        tempSum = 0
        for i in range(34, 36):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 8

        #生态环境与资源学院
        tempSum = 0
        for i in range(36, 37):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 10

        #管理学院
        tempSum = 0
        for i in range(37, 41):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 94

        #数学与统计学院
        tempSum = 0
        for i in range(41, 42):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 26

        #法学院
        tempSum = 0
        for i in range(42, 43):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 8
        #经济学院
        tempSum = 0
        for i in range(43, 47):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 40

        #艺术与设计学院
        tempSum = 0
        for i in range(47, 49):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 0

        #建筑与城市规划规划学院
        tempSum = 0
        for i in range(49, 52):
            for j in range(1, col):
                tempSum += lpvars[i][j]
            pro += tempSum <= 59

        #限制指派矩阵里面数量
        #优势学科只能取值0,2
        #5,6,7,8,13,20,28,30,34,35
        for i in (5,6,7,8,13,20,28,30,34,35):
            for j in range(1,col):
                pro += pl.lpSum(binary_variables[i, j, k] for k in [0, 2]) == 1
        # # #
        # # # for i in range(9):
        # # #     countcut = 0
        # # #     for j in range(0, col):
        # # #         countcut += binary_variables[i, j, 3]
        # # #     pro += countcut <= 1
        # #
        # #
        for i in (5,6,7,8,13,20,28,30,34,35):
            for j in range(1,col):
                pro += lpvars[i][j] == 0 * binary_variables[i, j, 0] + 2 * binary_variables[i, j, 2]
        #
        #
        #个别省份要分散
        for j in (3,5,6,7,9,10,11,14,16,17,18,19,20,23,24):
            for i in range(row):
                pro += pl.lpSum(binary_variables[i, j, k] for k in [0, 2, 3]) == 1

        for j in (3,5,6,7,9,10,11,14,16,17,18,19,20,23,24):
            for i in range(row):
                pro += lpvars[i][j] == 0 * binary_variables[i, j, 0] + 2 * binary_variables[i, j, 2] + 3 * binary_variables[i, j, 3]
        #
        #
        #
        for j in (2,15,21,22):
            for i in range(row):
                pro += pl.lpSum(binary_variables[i, j, k] for k in [0, 2]) == 1

        for j in (2,15,21,22):
            for i in range(row):
                pro += lpvars[i][j] == 0 * binary_variables[i, j, 0] + 2 * binary_variables[i, j, 2]


        for j in (1,4,8):
            for i in range(row):
                pro += pl.lpSum(binary_variables[i, j, k] for k in [0, 2, 3,4]) == 1

        for j in (1,4,8):
            for i in range(row):
                pro += lpvars[i][j] == 0 * binary_variables[i, j, 0] + 2 * binary_variables[i, j, 2] + 3 * binary_variables[i, j, 3] + 4 * binary_variables[i, j, 4]

        #物联网工程和大数据不出省
        tempSum = 0
        for j in range(1,col):
            tempSum += lpvars[7][j]
            tempSum += lpvars[8][j]
        pro += tempSum == 0



        # for i in range(9, row):
        #     for j in range(col):
        #         pro += pl.lpSum(binary_variables[i, j, k] for k in [0, 2, 3, 4]) == 1
        #
        # for i in range(9, row):
        #     for j in range(col):
        #         pro += lpvars[i][j] == 0 * binary_variables[i, j, 0] + 2 * binary_variables[i, j, 2] + 3 * \
        #                    binary_variables[i, j, 3] + 4 * binary_variables[i, j, 4]


        #每个省份都必须要有优势学科数量超过10%
        for j in range(1,col):
            tempSum = 0
            for i in (5,6,7,8,13,20,28,30,34,35):
                tempSum += lpvars[i][j]
            pro += tempSum >= L[j] * 0.1
        #省内计划留10
        for i in range(row):
            pro += lpvars[i][0] >= 10



        #尝试增加省内好专业数量
        tempSum = 0
        for i in range(row):
            tempSum += lpvars[i][0] * Q[i][0]
        pro += tempSum >= 2800

        c = cplex.Cplex()
        start_time = c.get_time()
        #status = pro.solve(pl.CPLEX_CMD(timeLimit=120))
        status = pro.solve(pl.CPLEX_CMD(timeLimit=120))
        end_time = c.get_time()
        print("Total solve time (sec.):", end_time - start_time)
        print("工作分配状态： ", pl.LpStatus[status])
        print("最终分配结果: ", pl.value(pro.objective))


        # get the result of T matrix
        T = [[lpvars[i][j].varValue
                for j in range(col)] for i in range(row)]
        #print(T)
        df = pd.DataFrame(T)
        excel_file_path = 'outputgdut2024.5.30V2.1.xlsx'
        df.to_excel(excel_file_path, index=False)
        return T




if __name__ == '__main__':
    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='Q')
    Q = df_sheet1.values
    #print(Q)
    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='L')
    L1 = df_sheet1.values
    L = [0.0 for i in range(len(L1[0]))]
    for i in range(len(L1[0])):
        L[i] = L1[0][i]
    #print(L)
    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='La')
    L1 = df_sheet1.values
    La = [0 for i in range(len(L1[0]))]
    for i in range(len(L1[0])):
        La[i] = L1[0][i]

    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='La2')
    L1 = df_sheet1.values
    La2 = [0 for i in range(len(L1[0]))]
    for i in range(len(L1[0])):
        La2[i] = L1[0][i]

    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='Sl')
    L1 = df_sheet1.values
    Sl = [0 for i in range(len(L1[0]))]
    for i in range(len(L1[0])):
        Sl[i] = L1[0][i]

    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='2023T')
    T2023 = df_sheet1.values

    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='2024T')
    T2024 = df_sheet1.values

    df_sheet1 = pd.read_excel('2024Qgdut.xlsx', sheet_name='jcT')
    Tjc = df_sheet1.values



    T = np.array(Assignment.KM(Q, La, L,Sl,T2023,T2024,Tjc,La2))


