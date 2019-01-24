#!/usr/bin/python
# coding:utf-8
import numpy as np
import scipy.sparse as ss
import os


class Deepwalk:

    # e->expert p->project d->discipline ci->discipline type
    def __init__(self, enum, pnum, dnum, dtnum, train_rate):
        train_rate = str(train_rate)

        # TODO 为何要加1？
        self.enum = enum + 1
        self.pnum = pnum + 1
        self.dnum = dnum + 1
        self.dtnum = dtnum + 1
        self.train_rate = train_rate

    def run(self):

        train_rate = self.train_rate

        # 加载训练数据
        ep = self.load_ep('../data/ep_' + train_rate + '.train')

        # 生成元路径，相关元路径的邻接矩阵相乘
        self.get_epe(ep, '../data/metapath/epe_' + train_rate + '_dpwk.txt')
        self.get_epdpe(ep, '../data/pd.txt', '../data/metapath/epdpe_' + train_rate + '_dpwk.txt')
        self.get_epdtpe(ep, '../data/pdt.txt', '../data/metapath/epdtpe_' + train_rate + '_dpwk.txt')
        self.get_pep(ep, '../data/metapath/pep_' + train_rate + '_dpwk.txt')
        self.get_pdtp('../data/pdt.txt', '../data/metapath/pdtp_' + train_rate + '_dpwk.txt')
        self.get_pdp('../data/pd.txt', '../data/metapath/pdp_' + train_rate + '_dpwk.txt')
        self.gen_embedding(train_rate)

    # 1.将e-p矩阵中的值初始化为0
    # 2.从文件中读取e-p的评分数据，矩阵对应位置置为1
    def load_ep(self, ubfile, weighted=False):
        ub = np.zeros((self.enum, self.pnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                # TODO 为何要将评分置为1
                if weighted:
                    ub[int(user)][int(item)] = rating
                else:
                    ub[int(user)][int(item)] = 1
        # 转化成适合处理稀疏矩阵的数据格式
        ub = ss.csc_matrix(ub)
        return ub

    def get_epe(self, ub, targetfile):
        print('EPE adjacency matrix multiplication ...')
        uu = ub.dot(ub.T)
        uu = self.sparse2dense(uu)
        print(uu.shape)

        print('writing to file...')
        self.save(targetfile, uu)

    def get_pep(self, ub, targetfile):
        print('PEP adjacency matrix multiplication...')
        mm = ub.T.dot(ub)
        mm = self.sparse2dense(mm)
        print(mm.shape)

        print('writing to file...')
        self.save(targetfile, mm)

    def get_pdtp(self, bcifile, targetfile):
        print('PDtP adjacency matrix initialization..')
        bci = self.matrix_init(bcifile, self.pnum, self.dtnum)

        print('PDtP adjacency matrix multiplication..')
        mm = bci.dot(bci.T)
        mm = self.sparse2dense(mm)

        print('writing to file...')
        self.save(targetfile, mm)

    def get_pdp(self, bcafile, targetfile):
        print('PDP adjacency matrix initialization..')
        bca = self.matrix_init(bcafile, self.pnum, self.dnum)

        print('PDP adjacency matrix multiplication..')
        mm = bca.dot(bca.T)
        mm = self.sparse2dense(mm)

        print('writing to file...')
        self.save(targetfile, mm)

    def get_epdpe(self, ub, bcafile, targetfile):
        print('EPDPE adjacency matrix initialization..')
        bca = self.matrix_init(bcafile, self.pnum, self.dnum)

        print('EPDPE adjacency matrix multiplication...')
        uu = ub.dot(bca).dot(bca.T).dot(ub.T)
        uu = self.sparse2dense(uu)

        print('writing to file...')
        self.save(targetfile, uu)

    def get_epdtpe(self, ub, bcifile, targetfile):
        print('EPDtPE adjacency matrix initialization..')
        bci = self.matrix_init(bcifile, self.pnum, self.dtnum)

        print('EPDtPE adjacency matrix multiplication...')
        uu = ub.dot(bci).dot(bci.T).dot(ub.T)
        uu = self.sparse2dense(uu)

        print('writing to file...')
        self.save(targetfile, uu)

    def sparse2dense(self, matrix):
        # np.save('../data/metapath/' + filename, matrix)
        # matrix = np.load('../data/metapath/' + filename + '.npy')[()]
        matrix = matrix.toarray()
        return matrix

    def matrix_init(self, file, row_num, colomn_num):
        matrix = np.zeros((row_num, colomn_num))
        with open(file, 'r') as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                matrix[int(m)][int(d)] = 1
        sparse_matrix = ss.csc_matrix(matrix)
        return sparse_matrix

    def save(self, targetfile, matrix):
        total = 0
        with open(targetfile, 'w') as outfile:
            rows, cols, data = ss.find(matrix)
            for i in range(len(rows)):
                outfile.write(str(rows[i]) + '\t' + str(cols[i]) + '\t' + str(data[i]) + '\n')
                total += 1
        print('total = ', total)

    def gen_embedding(self, train_rate):
        dim = 128
        walk_len = 10
        win_size = 3
        num_walk = 5

        metapaths = ['epe', 'epdpe', 'epdtpe', 'pep', 'pdp', 'pdtp']

        for metapath in metapaths:
            metapath = metapath + '_' + str(train_rate) + '_dpwk.txt'
            input_file = '../data/metapath/' + metapath
            output_file = '../data/embedding/' + metapath

            cmd = 'deepwalk --format edgelist --input ' + input_file + ' --output ' + output_file + \
                  ' --walk-length ' + str(walk_len) + ' --window-size ' + str(win_size) + ' --number-walks ' \
                  + str(num_walk) + ' --representation-size ' + str(dim)

            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    Deepwalk(enum=31868, pnum=43286, dnum=3541, dtnum=28, train_rate=0.9).run()

