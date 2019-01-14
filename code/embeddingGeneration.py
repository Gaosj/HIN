import os

train_rate = 0.9
dim = 128
walk_len = 10
win_size = 3
num_walk = 5

metapaths = ['epe', 'epdpe', 'epdtpe', 'pep', 'pdp', 'pdtp']

for metapath in metapaths:
    metapath = metapath + '_' + str(train_rate) + '.txt'
    input_file = '../data/metapath/' + metapath
    output_file = '../data/embedding/' + metapath

    cmd = 'deepwalk --format edgelist --input ' + input_file + ' --output ' + output_file + \
          ' --walk-length ' + str(walk_len) + ' --window-size ' + str(win_size) + ' --number-walks '\
           + str(num_walk) + ' --representation-size ' + str(dim)

    print(cmd)
    os.system(cmd)

