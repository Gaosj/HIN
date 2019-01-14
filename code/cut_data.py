#!/user/bin/python
import random

train_rate = 0.9

R = []
# 读取文档数据
with open('../data/ep.txt', 'r') as infile:
    for line in infile.readlines():
        print(line.strip().split('\t'))
        user, item, rating = line.strip().split('\t')
        R.append([user, item, rating])
# 数据打乱
random.shuffle(R)
# train_rate？
train_num = int(len(R) * train_rate)


with open('../data/ep_' + str(train_rate) + '.train', 'w') as trainfile, \
        open('../data/ep_' + str(train_rate) + '.test', 'w') as testfile:
    for r in R[:train_num]:
        trainfile.write('\t'.join(r) + '\n')
    for r in R[train_num:]:
        testfile.write('\t'.join(r) + '\n')
