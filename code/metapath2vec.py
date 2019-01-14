import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from stellargraph.data.loader import load_dataset_SMDB
from stellargraph.data.loader import load_dataset_BlogCatalog3
from gensim.models import Word2Vec
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph

location = "C:\\Users\\csdc01\\PycharmProjects\\demo\\HGRecDemo\\data\\"

# infos = [['e.csv', 'node', 'e'],
#          ['p.csv', 'node', 'p'],
#          ['d.csv', 'node', 'd'],
#          ['dt.csv', 'node', 'dt'],
#          ['e-p.csv', 'edge', 'belongs1', 'e', 'p'],
#          ['p-d.csv', 'edge', 'belongs2', 'p', 'd'],
#          ['p-dt.csv', 'edge', 'belongs3', 'p', 'dt']]
infos = [['e.csv', 'node', 'e'],
         ['p.csv', 'node', 'p'],
         ['e-p.csv', 'edge', 'belongs1', 'e', 'p']]

metapaths = [
    ['p', 'e', 'p']
]


def embedding(metapath):
    walks = rw.run(nodes=list(g_nx.nodes()),  # root nodes
                   length=100,  # maximum length of a random walk
                   n=10,  # number of random walks per root node
                   metapaths=metapath  # the metapaths
                   )
    print("Number of random walks: {}".format(len(walks)))

    model = Word2Vec(walks, size=128, window=3, min_count=0, sg=1, workers=4, iter=5)
    # model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=2, iter=1)

    save('../data/embedding/p-e-p.txt', model.wv)


def save(targetfile, word_vec):
    total = 0
    with open(targetfile, 'w') as outfile:
        outfile.writelines(str(len(word_vec.vectors)) + ' 128\n')
        for entity in word_vec.index2entity:
            string = str(word_vec.get_vector(entity))
            string = string.replace('[', '').replace(']', '').replace('\n', '').replace('  ', ' ')
            if string[:1] is '-':
                string = ' ' + string
            entity = str(entity).replace('e', '')
            string = string.replace('  ', ' ')
            string = string.replace('  ', ' ')
            outfile.writelines(entity + string + '\n')
            total += 1


if __name__ == '__main__':
    g_nx = load_dataset_SMDB(location, infos)
    print("Number of nodes {} and number of edges {} in graph.".format(g_nx.number_of_nodes(), g_nx.number_of_edges()))
    rw = UniformRandomMetaPathWalk(StellarGraph(g_nx))
    for metapath in metapaths:
        embedding(metapaths)
# print(model.wv.vectors.shape)
#
#
# # Visualise Node Embeddings
# # Retrieve node embeddings and corresponding subjects
# node_ids = model.wv.index2word  # list of node IDs
# node_embeddings = model.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
# node_targets = [ g_nx.node[node_id]['label'] for node_id in node_ids]
#
# # Transform the embeddings to 2d space for visualisation
# transform = TSNE #PCA
#
# trans = transform(n_components=2)
# node_embeddings_2d = trans.fit_transform(node_embeddings)
#
# # draw the points
# label_map = { l: i for i, l in enumerate(np.unique(node_targets))}
# node_colours = [ label_map[target] for target in node_targets]
#
# plt.figure(figsize=(20,16))
# plt.axes().set(aspect="equal")
# plt.scatter(node_embeddings_2d[:,0],
#             node_embeddings_2d[:,1],
#             c=node_colours, alpha=0.2)
# plt.title('{} visualization of node embeddings'.format(transform.__name__))
# plt.show()
