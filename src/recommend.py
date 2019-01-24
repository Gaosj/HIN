import numpy as np
import scipy.sparse as ss
import pandas as pd


def load():
    u = np.loadtxt('../data/predict_U.txt')
    e = np.loadtxt('../data/predict_E.txt')
    v = np.loadtxt('../data/predict_V.txt')
    h = np.loadtxt('../data/predict_H.txt')
    return u, e, v, h


def recommend():
    # 记得修改P的数量
    ep = matrix_init('../data/ep_predict.txt', row_num=31869, colomn_num=43286)
    # pd = matrix_init('../data/pd.txt', row_num=55702, colomn_num=3542)
    ee = ep.dot(ep.T)
    save('../data/ee_predict.txt', ee)
    # ed = ep.dot(pd)
    # save('../data/de_predict.txt', ed.T)


def matrix_init(file, row_num, colomn_num):
    matrix = np.zeros((row_num, colomn_num))
    with open(file, 'r') as infile:
        for line in infile.readlines():
            m, d, _ = line.strip().split('\t')
            matrix[int(m)][int(d)] = _
    sparse_matrix = ss.csc_matrix(matrix)
    return sparse_matrix


def save(targetfile, matrix):
    total = 0
    with open(targetfile, 'w') as outfile:
        rows, cols, data = ss.find(matrix)
        for i in range(len(rows)):
            # if(rows[i] is not cols[i])
            outfile.write(str(rows[i]) + '\t' + str(cols[i]) + '\t' + str(data[i]) + '\n')
            total += 1
    print('total = ', total)


def cal_N():
    df = pd.read_csv(filepath_or_buffer='../data/ee_predict.csv', index_col=0)
    print(df.shape[1])
    print(df.ix[27])


if __name__ == '__main__':
    cal_N()
