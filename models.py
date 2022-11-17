import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, NSModule


class NSGCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(NSGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 16)  # 16 or 18
        self.adj1 = NSModule(nfeat, 16)
        # self.gc2 = GraphConvolution(25*2, 20)
        # self.adj2 = adj_Features(25*2, 20)
        # self.gc4 = GraphConvolution(20 * 2, 15)
        # self.adj4 = adj_Features(20 * 2, 15)
        # self.gc5 = GraphConvolution(15 * 2, 7)
        # self.adj5 = adj_Features(15 * 2, 7)
        self.gc3 = GraphConvolution(16 * 2, nclass)
        self.dropout = dropout

    def forward(self, x, adj, adj_weight):
        x1 = F.relu(self.gc1(x, adj, adj_weight))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.adj1(x, adj, adj_weight))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x = torch.cat((x1, x2), 1)

        # x1 = F.relu(self.gc2(x, adj, adj_weight))
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x2 = F.relu(self.adj2(x, adj, adj_weight))
        # x2 = F.dropout(x2, self.dropout, training=self.training)
        # x = torch.cat((x1, x2), 1)
        #
        # x1 = F.relu(self.gc4(x, adj, adj_weight))
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x2 = F.relu(self.adj4(x, adj, adj_weight))
        # x2 = F.dropout(x2, self.dropout, training=self.training)
        # x = torch.cat((x1, x2), 1)
        #
        # x1 = F.relu(self.gc5(x, adj, adj_weight))
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x2 = F.relu(self.adj5(x, adj, adj_weight))
        # x2 = F.dropout(x2, self.dropout, training=self.training)
        # x = torch.cat((x1, x2), 1)

        x = self.gc3(x, adj, adj_weight)
        return F.log_softmax(x, dim=1)
