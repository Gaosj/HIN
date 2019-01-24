#!/usr/bin/python
# encoding=utf-8
from math import sqrt, fabs

import numpy as np

from src.data_process import input_gen
from src.data_process import get_metapaths
from src.metapath2vec import Metapath2vec
from src.deepwalk import Deepwalk


class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, embedding, train_rate):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)

        self.steps = 100
        self.delta = 0.02
        self.beta_e = 0.1
        self.beta_h = 0.1
        self.beta_p = 2
        self.beta_w = 0.1
        self.beta_b = 0.1
        self.reg_u = 1.0
        self.reg_v = 1.0

        self.train_file = '../data/ep_' + str(train_rate) + '.train'
        self.test_file = '../data/ep_' + str(train_rate) + '.test'
        self.embedding = embedding
        self.train_rate = train_rate

        self.user_metapaths, self.item_metapaths, self.user_metapathnum, self.item_metapathnum = get_metapaths(train_rate, embedding)

        self.X = None
        self.Y = None
        self.R = None
        self.T = None
        self.user_metapathdims = None
        self.item_metapathdims = None

    def run(self):
        # input_gen(self.train_rate)
        #
        if self.embedding is 'dpwk':
            Deepwalk(enum=self.unum, pnum=self.inum, dnum=3541, dtnum=28, train_rate=0.9).run()

        if self.embedding is 'mp2vec':
            Metapath2vec(self.train_rate).run()

        print('Start load embedding.')
        self.X, self.user_metapathdims = self.load_embedding(self.user_metapaths, self.unum)
        self.Y, self.item_metapathdims = self.load_embedding(self.item_metapaths, self.inum)
        print('Load embedding finished.')

        self.R, self.T = self.load_rating(self.train_file, self.test_file)

        print('max_steps: ', self.steps)
        print('delta: ', self.delta, 'beta_e: ', self.beta_e, 'beta_h: ', self.beta_h, 'beta_p: ', self.beta_p,
              'beta_w: ', self.beta_w, 'beta_b', self.beta_b, 'reg_u', self.reg_u, 'reg_v', self.reg_v)
        print('Load rating finished.')
        print('train size : ', len(self.R))
        print('test size : ', len(self.T))

        self.initialize()
        self.recommend()
        self.get_prediction()

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapath_dims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embedding/' + metapath
            print('Loading embedding data, location: %s' % sourcefile)
            with open(sourcefile) as infile:

                k = int(infile.readline().strip().split(' ')[1])
                print('Metapath: %s. The dim of metapath embedding: %d' % (metapath, k))
                metapath_dims.append(k)

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
        return X, metapath_dims

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

            with open('../data/predict_Wu.txt', 'w') as outfile:
                for data in self.Wu[0]:
                    outfile.write(str(data) + '\n')
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
        for step in range(self.steps):
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

    def get_prediction(self):
        print('get rating matrix...')
        np.savetxt('../data/predict_U.txt', self.U)
        np.savetxt('../data/predict_V.txt', self.V)
        np.savetxt('../data/predict_H.txt', self.H)
        np.savetxt('../data/predict_E.txt', self.E)
        np.savetxt('../data/predict_pu.txt', self.pu)
        np.savetxt('../data/predict_pv.txt', self.pv)
        # np.savetxt('../data/predict_Wu.txt', self.Wu)
        self.save_multi_matrix(self.Wu, self.user_metapathnum, "Wu")
        # np.savetxt('../data/predict_Wv.txt', self.Wv)
        self.save_multi_matrix(self.Wv, self.item_metapathnum, "Wv")
        # np.savetxt('../data/predict_bu.txt', self.bu)
        # np.savetxt('../data/predict_bv.txt', self.bv)
        self.save_multi_matrix(self.bu, self.user_metapathnum, "bu")
        self.save_multi_matrix(self.bv, self.item_metapathnum, "bv")

    def save_multi_matrix(self, matrix, matrix_num, matrix_name):
        for i in range(matrix_num):
            np.save('../data/predict_' + matrix_name + '_' + str(i) + '.txt', matrix[i])

    def save(self, targetfile, matrix):
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(matrix.shape[0])[1:]:
                for j in range(matrix.shape[1])[1:]:
                    if matrix[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
                        total += 1
        print('total = ', total)


if __name__ == "__main__":

    deepwalk = 'dpwk'
    mp2vec = 'mp2vec'

    hnrec = HNERec(unum=31868, inum=43286, ratedim=10, userdim=15, itemdim=10, embedding=mp2vec, train_rate=0.9)
    hnrec.run()
