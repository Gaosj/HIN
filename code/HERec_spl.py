#!/usr/bin/python
# encoding=utf-8
import numpy as np
import time
import random
import traceback
from math import sqrt, fabs, log, exp
import sys


class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile,
                 steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u
        self.reg_v = reg_v

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
        # print('Load user embedding finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        # print('Load user embedding finished.')

        self.R, self.T = self.load_rating(trainfile, testfile)
        print('Load rating finished.')
        print('train size : ', len(self.R))
        print('test size : ', len(self.T))

        self.initialize()
        self.recommend()

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embedding/' + metapath
            print('Loading embedding data, location: %s' % sourcefile)
            with open(sourcefile) as infile:

                k = int(infile.readline().strip().split(' ')[1])
                print('Metapath: %s. The dim of metapath embedding: %d' % (metapath, k))
                metapathdims.append(k)

                # 根据不同的元路径，创建一个二维数组.
                # 数组的第二维度为 Expert/Project 的特征空间的表示 row=Expert/Project col=feature(1,..,k)
                for i in range(num):
                    # 第i个Expert/Project，在当前metapath下的特征空间的表示
                    X[i][ctn] = np.zeros(k)

                for line in infile.readlines():
                    # 获取特征空间向量中每个维度的值
                    arr = line.strip().split(' ')
                    # 将序号转成index
                    i = int(arr[0]) - 1
                    # 将每个维度值附给 X[i][ctn][j]
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        # 从源数据中分割了 百分之train_rate的数据，作为真实值，在分割之前打乱了数据顺序
        r_train = []
        r_test = []
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                rating = float(rating)
                r_train.append([int(user) - 1, int(item) - 1, int(rating)])
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                rating = float(rating)
                r_test.append([int(user) - 1, int(item) - 1, int(rating)])
        return r_train, r_test

    def initialize(self):
        # 利用正态分布填充一个 row_num=unum col_num=itemdim 的矩阵
        # unum X itemdim
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        # inum X userdim
        self.H = np.random.randn(self.inum, self.userdim) * 0.1
        # unum X ratedim
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1
        # inum X ratedim
        self.V = np.random.randn(self.inum, self.ratedim) * 0.1

        # unum X 3
        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        # inum X 3
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum

        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            # userdim X 128
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            # userdim X 1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            # itemdim X 128
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            # itemdim X 1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    def sigmod(self, x):
        # Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间
        return 1 / (1 + np.exp(-x))

    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            # 将生成的 userdim X 128（由正态分布的值填充）与某专家的embedding 128 X 1 做点乘，再加上 userdim X 1（由正态分布填充的矩阵）
            # s3 最后的维度是 userdim X 1
            # TODO 为何要使用Sigmoid 函数？
            s3 = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])
            # ui 最终为 s3 乘以每条元路径的权重，通过这样的方式，将所有元路径的embedding融合在一起？
            ui += self.pu[i][k] * s3
        return self.sigmod(ui)

    def cal_v(self, j):
        # 原理同cal_u
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * self.sigmod((self.Wv[k].dot(self.Y[j][k]) + self.bv[k]))
        return self.sigmod(vj)

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj)

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = self.get_rating(i, j)

            if r_p > 5: r_p = 5
            if r_p < 1: r_p = 1
            m = fabs(r_p - r)
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self):
        mae = []
        rmse = []
        perror = 99999
        cerror = 9999
        n = len(self.R)
        s = 0
        for step in range(steps):
            total_error = 0.0
            for t in self.R:
                # 用作训练的e-id
                i = t[0]
                # 用作训练的p-id
                j = t[1]
                # 用作训练的rating值
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij

                # SGD优化
                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]
                self.U[i, :] -= self.delta * U_g
                self.V[j, :] -= self.delta * V_g

                ui = self.cal_u(i)
                for k in range(self.user_metapathnum):
                    x_t = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])

                    pu_g = self.reg_u * -eij * (ui * (1 - ui) * self.H[j, :]).dot(x_t) + self.beta_p * self.pu[i][k]

                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array(
                        [ui * (1 - ui) * x_t * (1 - x_t) * self.H[j, :]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    bu_g = self.reg_u * -eij * ui * (1 - ui) * self.pu[i][k] * self.H[j, :] * x_t * (
                            1 - x_t) + self.beta_b * self.bu[k]
                    # print pu_g
                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    y_t = self.sigmod(self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
                    pv_g = self.reg_v * -eij * (vj * (1 - vj) * self.E[i, :]).dot(y_t) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij * self.pv[j][k] * np.array(
                        [vj * (1 - vj) * y_t * (1 - y_t) * self.E[i, :]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * vj * (1 - vj) * self.pv[j][k] * self.E[i, :] * y_t * (
                            1 - y_t) + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

            perror = cerror
            cerror = total_error / n

            self.delta = 0.93 * self.delta

            if abs(perror - cerror) < 0.0001:
                s += 1
                break
            # print 'step ', step, 'crror : ', sqrt(cerror)
            MAE, RMSE = self.maermse()
            mae.append(MAE)
            rmse.append(RMSE)
        print('MAE: ', min(mae), ' RMSE: ', min(rmse))
        print(s
              )
        print(len(mae))
        print(len(rmse))

    def SGD(data):
        # Learn the vectors p_u and q_i with SGD.
        # data is a dataset containing all ratings + some useful info (e.g. number of items/users). '
        n_factors = 10  # number of factors
        alpha = .01  # learning rate
        n_epochs = 10  # number of iteration of the SGD procedure # Randomly initialize the user and item factors.
        p = np.random.normal(0, .1, (data.n_users, n_factors))
        q = np.random.normal(0, .1, (data.n_items, n_factors))  # Optimization procedure
        for _ in range(n_epochs):
            for u, i, r_ui in data.all_ratings():
                err = r_ui - np.dot(p[u], q[i])  # Update vectors p_u and q_i
                p[u] += alpha * err * q[i]
                q[i] += alpha * err * p[u]


if __name__ == "__main__":
    unum = 31868
    inum = 24225
    ratedim = 10
    userdim = 15
    itemdim = [10]
    train_rate = 0.9  # sys.argv[1]

    user_metapaths = ['e-p-e', 'e-p-dt-p-e', 'e-p-d-p-e']
    item_metapaths = ['p-e-p', 'p-dt-p', 'p-d-p']
    for i in range(len(user_metapaths)):
        user_metapaths[i] += '.txt'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '.txt'

    # user_metapaths = ['epe', 'epdtpe', 'epdpe']
    # item_metapaths = ['pep', 'pdtp', 'pdp']
    # for i in range(len(user_metapaths)):
    #     user_metapaths[i] += '_' + str(train_rate) + '.txt'
    # for i in range(len(item_metapaths)):
    #     item_metapaths[i] += '_' + str(train_rate) + '.txt'

    # user_metapaths = ['ubu_' + str(train_rate) + '.embedding', 'ubcibu_'+str(train_rate)+'.embedding',
    # 'ubcabu_'+str(train_rate)+'.embedding']

    # item_metapaths = ['bub_'+str(train_rate)+'.embedding', 'bcib.embedding', 'bcab.embedding']

    trainfile = '../data/ep_' + str(train_rate) + '.train'
    testfile = '../data/ep_' + str(train_rate) + '.test'
    steps = 100
    delta = 0.02
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.1
    reg_u = 1.0
    reg_v = 1.0
    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b',
          beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    for ud in itemdim:
        print('itemDim:', ud)
        HNERec(unum, inum, ratedim, userdim, ud, user_metapaths, item_metapaths, trainfile, testfile, steps, delta,
               beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
