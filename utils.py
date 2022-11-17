import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import Delaunay
import geopandas as gpd
import os
import pandas as pd
import random
from scipy.spatial.distance import pdist, squareform


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def adjacency(path, taz_points):  # 由TAZ面生成的内部点表示
    # build graph
    print("Loading {} dataset...".format(taz_points))
    file = gpd.read_file(os.path.join(path, taz_points))
    point_XY = file.values[:, 2:4]
    point_size = point_XY.shape[0]
    tri = Delaunay(point_XY)
    relation = np.array(tri.simplices)
    adjacency = np.zeros((point_size, point_size))  # 生成point_size * point_size维数据
    for i in range(len(relation)):
        Num_1 = relation[i, 0]
        Num_2 = relation[i, 1]
        Num_3 = relation[i, 2]
        adjacency[Num_1, Num_2] = adjacency[Num_2, Num_1] = 1

        adjacency[Num_2, Num_3] = adjacency[Num_3, Num_2] = 1

        adjacency[Num_1, Num_3] = adjacency[Num_3, Num_1] = 1

    # adjacency转稀疏矩阵
    adjacency = sp.coo_matrix(adjacency)

    return adjacency


def data_Genearte(path, sv_features, poi_features, labels):
    print("Loading {} dataset...".format(sv_features))

    temp = []
    with open(os.path.join(path, sv_features), "r") as f:
        for line in f.readlines():
            temp.append(line)

    sv_features = pd.DataFrame(columns=['sv_1'], dtype=np.float64)
    for i in range(2, 65):
        sv_features['sv_{}'.format(i)] = ''

    for i in range(len(temp)):
        sv_features.loc[len(sv_features)] = list(map(float, temp[i].split("\t")))

    sv_features = pd.DataFrame(normalize(sv_features))
    # ----------------------------------------------------------------------------------------
    print("Loading {} dataset...".format(poi_features))

    temp = []
    with open(os.path.join(path, poi_features), "r") as f:
        for line in f.readlines():
            temp.append(line)

    poi_features = pd.DataFrame(columns=['poi_1'], dtype=np.float64)

    for i in range(2, 14):
        poi_features['poi_{}'.format(i)] = ''

    for i in range(len(temp)):
        poi_features.loc[len(poi_features)] = list(map(float, temp[i].split("	")))

    poi_features = pd.DataFrame(normalize(poi_features))
    # ----------------------------------------------------------------------------------------
    print("Loading {} dataset...".format(labels))

    temp = []
    with open(os.path.join(path, labels), "r") as f:
        for line in f.readlines():
            temp.append(int(line))

    labels = temp

    # ----------------------------------------------------------------------------------------
    result = pd.concat([sv_features, poi_features], axis=1)  # 按列合并poi_features和poi_features
    # result = poi_features
    # result = sv_features
    return result, labels


def load_data(path="./data", dataset_view="sv_features.txt", poi="features_poi.txt", labels="label.txt",
              taz_points="taz_point.shp"):
    idx_features_labels = data_Genearte(path, dataset_view, poi, labels)  # 暂存特征和labels
    features = np.array(idx_features_labels[0], dtype=np.dtype(np.float64))

    features = sp.csr_matrix(features, dtype=np.float64)
    labels = encode_onehot(idx_features_labels[1])

    # build graph
    # adj需为对称矩阵,该条件在adjacency()函数中生成
    adj = adjacency(path, taz_points)
    adj_ = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 分别对特征矩阵features和邻接矩阵adj做标准化,normalize()方法
    # features = normalize(features)

    # 标准化之前加入自环
    adj = normalize(adj_ + sp.eye(adj_.shape[0]))

    # 生成加入sim权重的adj
    adj_weight = adj_ + sp.eye(adj_.shape[0])

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和临界矩阵的tensor
    # idx_train = list(random.sample(range(0, adj.shape[0]), 1000))
    # idx_val = list(random.sample(range(0, adj.shape[0]), 1000))
    # idx_test = list(set(range(1455)).difference(set(idx_train)))
    # 将lables输出为array格式
    labels_array = np.where(labels)[1]

    # 将label_list转格式为enumerate,并继续为list
    label_1 = [i for i, x in enumerate(labels_array) if x == 0]
    label_2 = [i for i, x in enumerate(labels_array) if x == 1]
    label_3 = [i for i, x in enumerate(labels_array) if x == 2]
    label_4 = [i for i, x in enumerate(labels_array) if x == 3]
    # label_5 = [i for i, x in enumerate(labels_array) if x == 4]

    label_1 = random.sample(label_1, int(len(label_1) * 0.5))
    label_2 = random.sample(label_2, int(len(label_2) * 1))
    label_3 = random.sample(label_3, int(len(label_3) * 0.5))
    label_4 = random.sample(label_4, int(len(label_4) * 1))

    label_1234 = label_1 + label_2 + label_3 + label_4

    random.seed(random.randint(1, 2000))

    idx_train_1 = random.sample(label_1, len(label_1) - 20)
    idx_train_2 = random.sample(label_2, len(label_2) - 20)
    idx_train_3 = random.sample(label_3, len(label_3) - 20)
    idx_train_4 = random.sample(label_4, len(label_4) - 20)
    idx_train = idx_train_1 + idx_train_2 + idx_train_3 + idx_train_4

    random.seed(random.randint(1, 2000))
    idx_val_1 = random.sample(label_1, int(len(label_1) * 0.8))
    idx_val_2 = random.sample(label_2, int(len(label_2) * 0.8))
    idx_val_3 = random.sample(label_3, int(len(label_3) * 0.8))
    idx_val_4 = random.sample(label_4, int(len(label_4) * 0.8))
    idx_val = idx_val_1 + idx_val_2 + idx_val_3 + idx_val_4

    idx_test = list(set(label_1234).difference(set(idx_train)))
    # with open("idx_test.txt", 'r+') as f:
    #     np.savetxt(f, idx_test, delimiter="\n")
    # 输出格式tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()  # 输出a_0
    adj_weight = sparse_mx_to_torch_sparse_tensor(adj_weight).to_dense()

    # 加入sim邻接矩阵
    sim = similarity(features)
    adj_weight = np.mat(torch.mul(sim, adj_weight))
    adj_weight = sp.csr_matrix(adj_weight)
    adj_weight = sparse_mx_to_torch_sparse_tensor(adj_weight).to_dense()  # 输出a_1
    # adj_weight = torch.mul(adj_weight, adj)
    # adj = normalize(torch.mul(sim, adj))

    # adj = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(adj)).to_dense()
    # 输出格式tensor
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, adj_weight, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def similarity(x):  # 返回相似性矩阵
    """returns one matrix which is similarity matrix."""
    x.numpy()
    cosine = squareform(pdist(x, 'cosine'))
    sim = 1 - torch.from_numpy(cosine)

    return sim.float()
