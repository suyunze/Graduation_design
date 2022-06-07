# %%
import os
import time

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from image_build import query_target_stock_data

label_dict = {str(i[1]): 1 if i[7] > 0 else 0 for i in query_target_stock_data()}  # 0:负样本，1：正样本
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
batch_size = 50

image_list = os.listdir('./npy')
rsd_pn_dict, fm_dict, factor_dict, trad_dict = [], [], [], []
rsd_list, fm_list, factor_list, trad_list = [], [], [], []

for i in image_list:
    if i.endswith('fm.npy'):
        fm_list.append(i[:21])
    elif i.endswith('Rsd.npy'):
        rsd_list.append(i[:21])
    elif i.endswith('factor.npy'):
        factor_list.append(i[:21])
    elif i.endswith('trad.npy'):
        trad_list.append(i[:21])

交集 = list(set(fm_list) & set(rsd_list) & set(factor_list) & set(trad_list))
交集.sort()
for i in 交集:
    fm_dict.append((np.load("npy/" + i + "fm.npy"), (i[10:20], label_dict[i[10:20]])))
    rsd_pn_dict.append((np.load("npy/" + i + "pnLaplace.npy"), (i[10:20], label_dict[i[10:20]])))
    factor_dict.append((np.load("npy/" + i + "factor.npy"), (i[10:20], label_dict[i[10:20]])))
    trad_dict.append((np.load("npy/" + i + "trad.npy"), (i[10:20], label_dict[i[10:20]])))

nums = len(交集)
print("数据集大小  ", nums)

fm_dict.sort(key=lambda x: x[1][0])
rsd_pn_dict.sort(key=lambda x: x[1][0])
factor_dict.sort(key=lambda x: x[1][0])
trad_dict.sort(key=lambda x: x[1][0])


class Datasets(Dataset):
    def __init__(self, data0, data1, data2, data3, label):
        self.data0, self.data1, self.data2, self.data3, self.label = torch.tensor(data0).type(
            torch.FloatTensor), torch.tensor(
            data1).type(torch.FloatTensor), torch.tensor(data2).type(torch.FloatTensor), torch.tensor(data3).type(
            torch.FloatTensor), label

    def __getitem__(self, item):
        return self.data0[item], self.data1[item], self.data2[item], self.data3[item], self.label[item]

    def __len__(self):
        return len(self.data0)


# %%
class GC_CNN(nn.Module):
    def __init__(self):
        super(GC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 256, kernel_size=(4, 4), stride=4, padding=0, device=device)  # rsd_pn_Lacpace
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(4, 1), stride=(4, 1), padding=0, device=device)  # fm 64*107*4
        self.bn2 = nn.BatchNorm2d(num_features=1)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, device=device)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=4, stride=4, padding=0, device=device)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.layer = nn.Linear(512 * 14 * 14, 14)

        self.conv5 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, device=device)
        self.bn5 = nn.BatchNorm2d(num_features=8)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, device=device)
        self.bn6 = nn.BatchNorm2d(num_features=16)
        self.maxpool6 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.layer2 = nn.Linear(16 * 4 * 4, 14)

        self.conv7 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, device=device)
        self.bn7 = nn.BatchNorm2d(num_features=4)
        self.maxpool7 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0, device=device)
        self.bn8 = nn.BatchNorm2d(num_features=8)
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.layer3 = nn.Linear(8 * 4 * 4, 9)

        self.layer4 = nn.Linear(37, 2)

    def forward(self, x1, x2, x4, x3):
        x2 = F.relu_(self.bn1(self.conv1(x2)))  # rsd
        x1 = F.relu_(self.bn2(self.conv2(x1)))  # fm
        x = torch.matmul(x1, torch.transpose(x1, 2, 3))
        x = torch.matmul(x2, x)  # + Parameter(torch.FloatTensor(x2))
        x = F.dropout(x, 0.4)
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.dropout(x, p=0.25)
        x = x.view(-1, 14 * 14 * 512)
        x = self.layer(x)
        x = F.dropout(x, p=0.5)

        x3 = F.relu_(self.bn5(self.conv5(x3)))  # factor
        x3 = self.maxpool5(x3)
        x3 = F.relu_(self.bn6(self.conv6(x3)))
        x3 = self.maxpool6(x3)
        x3 = F.dropout(x3, p=0.25)
        x3 = x3.view(-1, 4 * 4 * 16)
        x3 = self.layer2(x3)
        x3 = F.dropout(x3, p=0.5)

        x4 = F.relu_(self.bn7(self.conv7(x4)))  # trad
        x4 = self.maxpool7(x4)
        x4 = F.relu_(self.bn8(self.conv8(x4)))
        x4 = self.maxpool8(x4)
        x4 = F.dropout(x4, p=0.25)
        x4 = x4.view(-1, 4 * 4 * 8)
        x4 = self.layer3(x4)
        x4 = F.dropout(x4, p=0.5)

        x = torch.cat((x, x3, x4), 1)
        x = self.layer4(x)
        x = F.softmax(x, dim=1)
        return x


# %%
a = [[i[0]] for i in fm_dict][:-1]
b = [i[0] for i in rsd_pn_dict][:-1]
c = [[i[0]] for i in factor_dict][:-1]
d = [[i[0][0] + i[0][1] + i[0][2] + i[0][3]] for i in trad_dict][:-1]
e = [i[1][1] for i in fm_dict][1:]
print(len(a), len(b), len(c), len(d), len(e))
dataloader = DataLoader(dataset=Datasets(a[:-200], b[:-200], c[:-200], d[:-200], e[:-200]), batch_size=batch_size,
                        shuffle=True)
test = DataLoader(dataset=Datasets(a[-200:], b[-200:], c[-200:], d[-200:], e[-200:]), batch_size=batch_size,
                  shuffle=True)
network = GC_CNN().to(device)
# network.load_state_dict(torch.load('model/model_tensor(0.8078).pth'))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.8)
size = len(dataloader.dataset)
arr_loss = {}
arr_acc = {}
start = time.time()
for epoch in range(50):
    print('epoch:', epoch)
    loss_list = []
    acc_list = []
    print("-------------------------------------------------------------------------------")
    for batch, (i, j, k, l, label) in enumerate(dataloader):
        i, j, k, l = i.to(device), j.to(device), k.to(device), l.to(device)
        output = network(i, j, k, l)
        accuracy = torch.sum(
            torch.eq(torch.tensor([0. if i >= 0.6 else 1. if j >= 0.6 else 2. for i, j in output]).float(),
                     label).float()) / len(label)
        print(output, label)
        input()
        loss = loss_fn(output, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch * len(output)
        loss_list.append(loss)
        acc_list.append(accuracy)
        print("epoch :", epoch, "accu : ", '%.2f' % accuracy, f"  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if torch.mean(torch.tensor(acc_list)) >= 0.7:
        torch.save(network.state_dict(), 'model/model_' + str(torch.mean(torch.tensor(acc_list))) + '.pth')
    arr_loss[epoch] = torch.mean(torch.tensor(loss_list))
    arr_acc[epoch] = torch.mean(torch.tensor(acc_list))
    print("-------------------------------------------------------------------------------")
end = time.time()
arr_loss_fig = go.Figure(data=[
    go.Scatter(x=list(arr_loss.keys()), y=list(arr_loss.values()), mode='lines+markers', name='loss',
               textposition='top center')])
arr_loss_fig.update_layout(title='loss trad', xaxis_title='epoch', yaxis_title='loss')
arr_acc_fig = go.Figure(data=[
    go.Scatter(x=list(arr_acc.keys()), y=list(arr_acc.values()), mode='lines+markers', name='acc',
               textposition='top center')])
arr_acc_fig.update_layout(title='acc trad', xaxis_title='epoch', yaxis_title='acc')
consume = end - start
print(f"consume time: {consume:.2f}s")
arr_loss_fig.write_image("img/loss.png")
arr_acc_fig.write_image("img/acc.png")



# %%
network.load_state_dict(torch.load('model/model_tensor(0.8760).pth'))
network.eval()
try:
    test_size = len(test.dataset)
    test_acc_list = []
    test_x = []
    true = []
    for batch, (i, j, k, l, label) in enumerate(test):
        i, j, k, l = i.to(device), j.to(device), k.to(device), l.to(device)
        output = network(i, j, k, l)
        true += [(int(0), int(k)) if i >= 0.6 else (int(1), int(k)) if j >= 0.6 else (int(2), int(k)) for (i, j), k in zip(output, label)]
        accuracy = torch.sum(
            torch.eq(torch.tensor([0. if i >= 0.6 else 1. if j >= 0.6 else 2. for i, j in output]).float(),
                     label).float()) / len(label)
        if accuracy != 0:
            test_x.append(batch)
            test_acc_list.append(accuracy)
        if batch != len(test)-1:
            print("test accu : ", '%.2f' % accuracy, f"  [{batch * len(output):>5d}/{test_size:>5d}]")
        else:
            print("test accu : ", '%.2f' % accuracy, f"  [{test_size:>5d}/{test_size:>5d}]")

    test_img = go.Figure(data=[go.Bar(x=test_x, y=test_acc_list, textposition='auto')])
    test_img.update_layout(title='test accuracy', xaxis_title='batch', yaxis_title='accuracy')
    test_img.write_image('img/test_acc_' + str(batch_size) + '.png')
except:
    pass
true_img = go.Figure()
for i in range(1, len(true)):
    if true[i][0] != true[i][1]:
        if true[i][0] == 1.:
            true_img.add_trace(go.Scatter(x=[i - 1, i], y=[true[i - 1][0], true[i][0]], mode='lines',
                                          line=dict(color='rgb(43,174,133)', ),textposition='bottom center'))
        elif true[i][0] == 0.:
            true_img.add_trace(go.Scatter(x=[i - 1, i], y=[true[i - 1][0], true[i][0]], mode='lines',
                                          line=dict(color='rgb(166,27,41)', ),textposition='bottom center'))
        else:
            true_img.add_trace(go.Scatter(x=[i - 1, i], y=[true[i - 1][0], true[i][0]], mode='lines',
                                          line=dict(color='rgb(254,186,7)', ),textposition='bottom center'))
true_img.update_layout(title="acc ture_false",xaxis_title="i(day)",yaxis_title="res")
true_img.write_image("img/acc_trad_true_false.png")
print(true)
# %%
