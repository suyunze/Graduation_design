import time
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset

from databaseORM import *

metadata_obj = MetaData()
mapper_registry = registry()
Base = mapper_registry.generate_base()
engine, session = db().content()
target_stock = '000004.SZ'


def query_target_stock_data():
    Item = item(target_stock)
    target_stock_data = []
    for i in session.query(Item).order_by(Item.date):
        # 查询目标股票list_up_year年内的交易数据
        # if int(str(i.date)[:4]) >= dt.datetime.today().year - list_up_year:
        target_stock_data.append(np.array([i.sp, i.udp]))
    target_stock_data = torch.Tensor(target_stock_data)
    return target_stock_data


target_data = query_target_stock_data()


# 创建 LSTM 层
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=64,  # 输入尺寸为 1，表示一天的数据
            hidden_size=64,
            num_layers=1,
            batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(64, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out = self.out(r_out[:, -1, :])  # 取最后一天作为输出
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        self.data = []
        self.label = []
        for i in data:
            self.data.append((i[:-1]).numpy().tolist())
            self.label.append((i[-1]).numpy().tolist())
        self.data, self.label = torch.Tensor(self.data), torch.Tensor(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


LR = 0.0001
EPOCH = 150
DAYS_BEFORE = 7
TEST = 200

rnn = torch.load('rnn.pkl').cuda()
rnn.train()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func =

print(target_data.shape)
data = (torch.amax(target_data[..., 0]) - target_data[..., 0]) / (
        torch.amax(target_data[..., 0]) - torch.amin(target_data[..., 0]))
max_min = torch.amax(target_data[..., 0]) - torch.amin(target_data[..., 0])
label = target_data[..., 1]
label_data = []
data_train = []
for i in range(0, len(data) - 67):
    data_train.append(data[i:i + 65])
    label_data.append(label[i + 66])

batch_size = 64

train, test = data_train[:-TEST], data_train[-TEST:]
train_set = TrainSet(train)
test_set = TrainSet(test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
start = time.time()

for step in range(EPOCH):
    for tx, ty in train_loader:

        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()
        output = rnn(torch.unsqueeze(tx, dim=1))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
    print("Epoch:", step, " loss: ", loss.item())
end = time.time()

rnn.eval()
pred_list = []
real_list = []
for tx, ty in test_loader:
    if torch.cuda.is_available():
        tx = tx.cuda()
        ty = ty.cuda()
    output = rnn(torch.unsqueeze(tx, dim=1))
    output = torch.squeeze(output)
    tx = tx[..., -1]
    pred = output * max_min - tx
    pred = torch.tensor([0 if i < 0 else 1 for i in pred])
    pred_list+=pred.tolist()
    real = ty - tx
    real = torch.tensor([0 if i < 0 else 1 for i in real])
    real_list+=real.tolist()
    acc = torch.sum(torch.eq(pred, real))/len(pred)
    print("acc: ", acc.item())
print([(i,j)for i,j in zip(pred_list, real_list)])
print("time: ", end - start)


torch.save(rnn, 'rnn.pkl')
