import itertools
import os
import datetime as dt

import numpy as np
import torch
from numba import cuda
import numpy_gpu as gnp
from sqlalchemy import *
from sqlalchemy.orm import registry

from databaseORM import db, item, Stock
from factors import *
from factors_cul import factors_cul

metadata_obj = MetaData()
mapper_registry = registry()
Base = mapper_registry.generate_base()
engine, session = db().content()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1、查询软件服务行业股票列表
stocklist = session.query(Stock).order_by(Stock.tscode).filter(Stock.industry == '软件服务').all()

# 2、设置目标股票为000004的预测
target_stock = '000004.SZ'


# 2、计算更新股票指标数据
def update_stock_factor():
    for i in stocklist:
        factors_cul(i.tscode).update_factors()
        print(i.tscode)


# 3、筛选成立时间大于8年的股票交易数据
def filterListUpEightStockData(list_up_year=12):
    max_date = {i: stocklist[i].tscode for i in range(0, len(stocklist)) if
                # dt.datetime.strptime(stocklist[i].listdate, '%Y%m%d') <= dt.datetime(2010, 1, 1) and
                stocklist[i].tscode != target_stock}  # 筛选成立日期不足8年的股票
    list_before_near_eight = []  # 创建成立日期大于8年的股票数据列表
    for i in max_date.keys():
        Item = item(stocklist[i].tscode)
        data = session.query(Item).order_by(Item.date)
        near_eight_year = {}
        for j in data:
            # if dt.datetime(j.date.year,j.date.month,j.date.day) >= dt.datetime(2008, 1, 1):  # dt.datetime.today().year - list_up_year:
            near_eight_year[j.date] = [j.code, j.date, j.bp, j.hp, j.lp, j.sp, j.yp, j.udp, j.udr, j.roc]
        list_before_near_eight.append(near_eight_year)
    return list_before_near_eight, max_date
    # list_before_near_eight =>
    #       [{date_1:data_1,date_2:data_2,...,date_n:data_n},
    #        {date_1:data_1,date_2:data_2,...,date_n:data_n},
    #        ...,
    #        {date_1:data_1,date_2:data_2,...,date_n:data_n}]
    # max_date => {index:tscode,...}


list_before_near_eight, max_date = filterListUpEightStockData()
stock_list = [target_stock] + [i.tscode for i in stocklist]


# # 计算rit值
# for i in stock_list+[target_stock]:
#     Item = item(i)
#     data = session.query(Item).order_by(Item.date).all()
#     mapping = rt([[i.date, i.sp] for i in data])
#     for j in data:
#         try:
#             j.rt = mapping[j.date]
#         except KeyError:
#             pass
#     session.flush()
#     session.commit()
#
# # 计算rit均值
#
#
# for i in stock_list + [target_stock]:
#     Item = item(i)
#     data = session.query(Item).order_by(Item.date).all()
#     mapping = rt_mean([[k.date, k.rt] for k in data if k.rt is not None])
#     for j in data:
#         try:
#             j.rta = mapping[j.date]
#         except KeyError:
#             pass
#     session.flush()
#     session.commit()[

# 4、创建相关度图像
def create_rsd_image(target=target_stock, rsd={}):
    rsd_data = {}
    for i, d in enumerate(stock_list):
        Item = item(d)
        data = session.query(Item).order_by(Item.date.desc())
        rsd_data[(i, d)] = {m.date: m.rt for m in data if m.rt is not None}  # {(,):[,,],...,(,):[,,]}
    for i in rsd_data.keys():
        m = {}
        keys = rsd_data[i].keys()
        for j in range(0, 6998):
            date_t = dt.date(2022, 2, 24) - dt.timedelta(j)
            if date_t not in keys:
                if date_t.isoweekday() != 6 and date_t.isoweekday() != 7:
                    if m != {}:
                        m[date_t] = m[list(m.keys())[-1]]
                    else:
                        m[date_t] = 0.0
                else:
                    ...
            elif date_t in keys:
                m[date_t] = rsd_data[i][date_t]
        rsd[i] = m
    print(len(m))
    return rsd


# create_rsd_image()

def create_rsd_image_all(lag, rsd_data=create_rsd_image(), target=target_stock):
    count = len(rsd_data[(0, target_stock)])
    if rsd_data is None:
        raise ValueError("无数据传入")
    else:
        print("第 " + str(lag) + "个开始 | total:  " + str((lag / count) * 100)[:6] + " %")
        img = np.zeros((len(stock_list), len(stock_list)))
        data = torch.tensor([list(i.values())[lag:count] for i in list(rsd_data.values())])  # [[,,,],[,,,],[,,,]...]
        for i, j in zip(list(itertools.combinations(range(0, len(data)), 2)),
                        list(itertools.combinations(data - data.mean(dim=1, keepdim=True), 2))):
            img_t = (j[0] * j[1]).sum() / ((j[0] ** 2).sum().sqrt() * (j[1] ** 2).sum().sqrt())
            if torch.isnan(img_t):
                img[i[0], i[1]] = 0.0
            else:
                img[i[0]][i[1]] = img_t  # 创建相关度矩阵
        img_f = img.T
        img = img + img_f
        np.fill_diagonal(img, 1)
        np.save("npy/" + target + "_" + str(list(list(rsd_data.values())[0].keys())[lag]) + "_Rsd.npy", img)
        print("第 " + str(lag) + "个完成 | total:  " + str((lag / count) * 100)[:6] + " %")


# create_rsd_image_all(0, create_rsd_image())


def Laplace(pos, neg):
    pos, neg = torch.Tensor(pos), torch.Tensor(neg)
    pos_D, neg_D = torch.zeros(pos.shape), torch.zeros(neg.shape)
    for i, j in enumerate(pos):
        if torch.sqrt(torch.sum(j)) != 0:
            pos_D[i][i] = 1 / torch.sqrt(torch.sum(j))
    for i, j in enumerate(neg):
        if torch.sqrt(torch.sum(j)) != 0:
            neg_D[i][i] = 1 / torch.sqrt(torch.sum(j))
    pos, neg = pos_D * pos * pos_D, neg_D * neg * neg_D
    return [pos, neg]


# 5、创建正负相关度图像
def create_rsd_image_pos_neg():
    pos, neg = [], []
    for k in os.listdir("./npy/"):
        if k.endswith("Rsd.npy"):
            img = np.load("npy/" + k)  # 读取相关度矩阵
            positive = np.zeros(img.shape)
            negative = np.zeros(img.shape)
            for i, m in enumerate(img):
                for j, n in enumerate(m):
                    if n >= 0:
                        positive[i][j] = n
                    elif n < 0:
                        negative[i][j] = abs(n)
            pos, neg = Laplace(positive, negative)
            a = np.array([np.array(pos), np.array(neg)])
            np.save("npy/" + k[:-7] + "pnLaplace.npy", a)  # 正相关矩阵保存


# positive = np.load("npy/"+"000004.SZ_positive.npy")  # 正相关矩阵读取
# negative = np.load("npy/"+"000004.SZ_negative.npy")  # 负相关矩阵读取


#
# Image.fromarray((positive / np.amax(positive)) * 255).show()  # 正相关矩阵显示
# Image.fromarray((negative / np.amax(negative) * 255)).show()  # 负相关矩阵显示


# 6、查询目标股票的交易数据
def query_target_stock_data(list_up_year=13):
    Item = item(target_stock)
    target_stock_data = []
    for i in session.query(Item).order_by(Item.date):
        # 查询目标股票list_up_year年内的交易数据
        # if int(str(i.date)[:4]) >= dt.datetime.today().year - list_up_year:
        target_stock_data.append([i.code, i.date, i.bp, i.hp, i.lp, i.sp, i.yp, i.udp, i.udr, i.roc])
    target_stock_data = np.array(target_stock_data)
    return target_stock_data
    # target_stock_data => ndarray
    # target_stock_data => [[i.code, i.date, i.bp, i.hp, i.lp, i.sp, i.yp, i.udp, i.udr],
    #                       [i.code, i.date, i.bp, i.hp, i.lp, i.sp, i.yp, i.udp, i.udr],
    #                       ...,
    #                       [i.code, i.date, i.bp, i.hp, i.lp, i.sp, i.yp, i.udp, i.udr]]


# 7、正负相关度矩阵归一化
def p_n_Normalization(pos, neg, target=target_stock):
    pos_nor = (pos - np.amin(pos)) / (np.amax(pos) - np.amin(pos))
    neg_nor = (neg - np.amin(pos)) / (np.amax(neg) - np.amin(pos))
    np.save("npy/" + target + "pos_nor.npy", pos_nor)
    np.save("npy/" + target + "neg_nor.npy", neg_nor)
    return pos_nor, neg_nor
    # [[[0.25,0.5,0.75,1],[0.25,0.5,0.75,1],..len_date..,[0.25,0.5,0.75,1]],
    #  [[0.25,0.5,0.75,1],[0.25,0.5,0.75,1],..len_date..,[0.25,0.5,0.75,1]],
    #  ...,
    #  [[0.25,0.5,0.75,1],[0.25,0.5,0.75,1],..len_date..,[0.25,0.5,0.75,1]]]
    #  fm => ndarray  (len_date, len_stock, 4)


# 9、创建特征矩阵序列
def create_feature_matrix(list_before_near_eight, target_stock_data=None, target=target_stock):
    if target_stock_data is None:
        target_stock_data = query_target_stock_data()
    for i in range(0, len(target_stock_data) - 1):
        print(str(i) + "  /   " + str(len(target_stock_data)))
        target_date = [j for j in target_stock_data[i:i + 8, 1]]
        fm_ = np.zeros((len(list_before_near_eight), 4))
        for k, t in enumerate(list_before_near_eight):
            file_path = "npy/" + target + "_" + str(target_date[-1]) + "_Rsd.npy"
            if os.path.exists(file_path):
                try:
                    roc_list = one_roc([list_before_near_eight[k][j][5] for j in target_date])
                except KeyError:
                    ...
                fm_[k][0] = np.load(file_path)[0][k]
                fm_[k][1] = roc_list[-1]
                fm_[k][2] = np.mean(roc_list)
                fm_[k][3] = np.std(roc_list)
        np.save("npy/" + target + "_" + str(target_date[-1]) + "_" + "fm.npy", fm_)
    # [[[1,2,3,4],[1,2,3,4],..len_date..,[1,2,3,4]],
    #  [[1,2,3,4],[1,2,3,4],..len_date..,[1,2,3,4]],
    #  ...,
    #  [[1,2,3,4],[1,2,3,4],..len_date..,[1,2,3,4]]]
    #  fm => ndarray  (len_date, len_stock, 4)


# 10、个股信息特征归一化*20
def individualStockInformation(target=target_stock):
    Item = item(target)
    data = session.query(Item).order_by(Item.date).all()
    data_list = {}
    for i in data:
        data_list[i.date] = [i.hp, i.bp, i.sp, i.lp]
    data_date, data = list(data_list.keys()), np.array(list(data_list.values()))
    return data_date, ((data - data.min(axis=0)) * 20) / (data.max(axis=0) - data.min(axis=0))
    # [[0.22035593 0.23618407 0.21936965 0.22608696]
    #  [0.22075585 0.22231561 0.23314548 0.2173913 ]
    #  [0.23455309 0.23954612 0.24754748 0.24782609]
    #  ...
    #  [0.3845231  0.40827905 0.3880192  0.40282609]
    #  [0.37732454 0.39378021 0.39261115 0.40195652]
    #  [0.37612478 0.38936751 0.37466082 0.38391304]]


# 10、生成个股信息矩阵图像
def individualStockInformationImage(isi, date):  # date,isi = individualStockInformation()
    for z, i in zip(date, range(0, len(isi))):
        # if i >= 4500:
        img = []
        for j in [0, 1, 2, 3]:
            channel = np.zeros((20, 20))
            for index, k in enumerate(isi[i:i + 20, j]):
                if math.ceil(k) <= 20:
                    channel[-math.ceil(k)][index] = 1 + math.modf(k)[0]
                else:
                    channel[0][index] = 2
            img.append(channel)
        np.save("npy/" + target_stock + "_" + str(z) + "_" + "trad.npy", np.array(img))

    # [[[[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]  # 20*20
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]]  # 4*20*20
    #
    #
    #  ...
    #
    #
    #  [[[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]](4980, 4, 20, 20)


# 11、查询目标股票技术指标
def technicalIndicators(target=target_stock):
    Item = item(target)
    data = session.query(Item).order_by(Item.date).all()
    data_list = {}
    for i in data:
        if i.cci != null and i.macd != null and i.wr and i.cmo != null and i.ema != null and i.hma != null and i.sma != null and i.wma != null and i.tema != null and i.roc != null:
            data_list[i.date] = [i.cci, i.macd, i.wr, i.cmo, i.ema, i.hma, i.sma, i.wma, i.tema, i.roc]
    return data_list


# 12、目标股票技术指标图像
def technicalIndicatorsImage(data=None):
    if data is None:
        data = technicalIndicators()
    imgs = []
    date = list(data.keys())
    data = np.array(list(data.values()))
    print(date, data)
    for i in range(0, len(data) - 20):
        img = np.zeros((20, 20))
        data_20 = data[i:i + 20, ...] * 2  # 取二十天十个指标 k=[[] [] [] ... []] α=2使区间[0,1]变为[0,2]
        for j in range(0, 10):
            for index_k, k in enumerate(data_20[..., j]):
                if k < 1:
                    img[(j * 2) + 1][index_k] = 1 + k
                elif 1 <= k <= 2:
                    img[(j * 2)][index_k] = k
        np.save("npy/" + target_stock + "_" + str(date[i]) + "_" + "factor.npy", np.array(img))
    return date, np.array(imgs)

    # [[[1.510158 1.457664 1.536552 ... 1.635636 1.797316 1.827308]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  [0.       0.       0.       ... 0.       1.033508 0.      ]
    #  ...
    #  [1.484462 1.482494 1.47822  ... 1.464712 1.500622 1.46405 ]
    #  [0.       0.       0.       ... 0.       0.       1.002618]
    #  [1.933992 1.902006 1.88863  ... 1.90317  1.956674 0.      ]]  # 20*20
    #
    # [[1.457664 1.536552 1.537258 ... 1.797316 1.827308 1.77391 ]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  [0.       0.       0.       ... 1.033508 0.       0.      ]
    #  ...
    #  [1.482494 1.47822  1.490422 ... 1.500622 1.46405  1.50981 ]
    #  [0.       0.       0.       ... 0.       1.002618 1.015992]
    #  [1.902006 1.88863  1.833964 ... 1.956674 0.       0.      ]]
    #
    # [[1.536552 1.537258 1.596994 ... 1.827308 1.77391  1.715884]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  ...
    #  [1.47822  1.490422 1.476328 ... 1.46405  1.50981  1.463834]
    #  [0.       0.       0.       ... 1.002618 1.015992 0.      ]
    #  [1.88863  1.833964 1.916546 ... 0.       0.       1.999128]]
    #
    # ...
    #
    # [[1.848834 1.907154 1.700034 ... 1.878    1.497562 1.47908 ]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  [1.050662 0.       0.       ... 0.       0.       0.      ]
    #  ...
    #  [1.844162 1.823874 1.84274  ... 1.855042 1.904424 1.859508]
    #  [1.027624 1.032276 1.005524 ... 1.042744 0.       0.      ]
    #  [0.       0.       0.       ... 0.       1.900262 1.915382]]
    #
    # [[1.907154 1.700034 1.546142 ... 1.497562 1.47908  1.513506]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  ...
    #  [1.823874 1.84274  1.8222   ... 1.904424 1.859508 1.884402]
    #  [1.032276 1.005524 0.       ... 0.       0.       0.      ]
    #  [0.       0.       1.948532 ... 1.900262 1.915382 1.86246 ]]
    #
    # [[1.700034 1.546142 1.475612 ... 1.47908  1.513506 1.447094]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  [0.       0.       0.       ... 0.       0.       0.      ]
    #  ...
    #  [1.84274  1.8222   1.844856 ... 1.859508 1.884402 1.858764]
    #  [1.005524 0.       0.       ... 0.       0.       0.      ]
    #  [0.       1.948532 1.919454 ... 1.915382 1.86246  1.868276]]] # (4972, 20, 20)

    # update_stock_factor()


def init(args):
    rsd_data = create_rsd_image()
    create_rsd_image_all(lag=args[0], rsd_data=rsd_data)


# print(b, b.shape)


# pos, neg = np.load("npy/"+target_stock+"_positive.npy"), np.load("npy/"+target_stock+"_negative.npy")
#
# lbne, max_date = filterListUpEightStockData()
# img = query_target_stock_data()

# igcn_train_data_1 = p_n_Normalization(pos, neg)


from multiprocessing import Pool

if __name__ == '__main__':
    count = item(stocklist[0].tscode)
    count = session.query(count).count()
    print(count)
    count_list = [i for i in range(0, count-5)]

    pool = Pool(processes=6)
    pool.map(create_rsd_image_all, count_list)
    pool.close()  # 关闭进程
    pool.join()

    create_rsd_image_pos_neg()
    create_feature_matrix(list_before_near_eight=list_before_near_eight)
    date, isi = individualStockInformation()
    individualStockInformationImage(isi, date)

    technicalIndicatorsImage(technicalIndicators())
