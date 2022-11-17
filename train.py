import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

sns.set_style('whitegrid')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch

torch.set_printoptions(profile="full")
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import NSGCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=4000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, adj_weight, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = NSGCN(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_weight = adj_weight.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, adj_weight)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss.append(loss_train.detach().tolist())

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, adj_weight)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    accuracy_list.append((acc_val.item()))
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj, adj_weight)
    y_pred.append(output[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
loss = []
accuracy_list = []
iter = range(args.epochs)
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

y_pred = []
# Testing
test()

# 绘制loss图像
color = cm.viridis(0.2)
f, ax = plt.subplots(1, 1)
ax.plot(iter, loss, label='loss', color=color)
ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('lOSS')
exp_dir = 'Loss/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
else:
    os.makedirs(exp_dir, exist_ok=True)
f.savefig(os.path.join('Loss', 'reward_{}'.format(args.epochs) + '.png'), dpi=600)


# with open("loss.txt", 'r+') as f:  # a+追加 r+覆盖
#     np.savetxt(f, loss, delimiter="\n")

# with open("accuracy.txt", 'r+') as f:  # a+追加 r+覆盖
#     np.savetxt(f, accuracy_list, delimiter="\n")


# 绘制混淆矩阵
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(8, 8), dpi=1000)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(labels.max().item() + 1)
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.4f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')


classes = []
for i in range(labels.max().item() + 1):
    classes.append("{}".format(i))
y_true = labels[idx_test].cpu().numpy()
# labels.cpu().numpy()  # 样本实际标签
y_pred = y_pred[0].cpu().detach().numpy()  # 样本预测标签
y_pred = np.argmax(y_pred, axis=1)
# with open("y_pred.txt", 'r+') as f:  # a+追加 r+覆盖
#     np.savetxt(f, y_pred, delimiter="\n")
# 获取混淆矩阵
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, os.path.join('./Loss', 'GCN-NS.png'.format(args.epochs)),
                      title='confusion_matrix_{}'.format('poi'))
# with open("confusion_matrix.txt", 'a') as f:  # a+追加 r+覆盖
#     np.savetxt(f, cm, delimiter="\n")
kappa = cohen_kappa_score(y_pred, y_true)
