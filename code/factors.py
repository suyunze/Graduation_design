# _*_ coding:utf-8 _*_
# main factors

import math
import numpy_gpu as gnp
import numpy as np
import torch


def CCI(dl):
    cci = {}
    date = dl[..., 1]
    for n in range(7, len(dl)):
        ma = 0
        md = 0
        cp = dl[n - 7:n, 5]
        for j in cp:
            ma = ma + j
        ma = ma / 7
        for k in cp:
            md = md + abs(ma - k)
        md = md / 7
        tp = 0
        for i in dl[n - 1, 3:6]:
            tp = tp + i
        tp = tp / 3
        cci[date[n]] = (tp - ma) / md / 0.015
    return cci


def WR(dl):
    wr = {}
    date = dl[..., 1]
    hp = dl[..., 3]
    lp = dl[..., 4]
    for i in range(7, len(hp)):
        j, k = max(hp[i - 7:i]), min(lp[i - 7:i])
        z = (j - dl[i, 5]) / (j - k) * 100
        wr[date[i]] = z
    return wr


def CMO(dl):
    cmo = {}
    date = dl[..., 1]
    yp = dl[1:, 7]
    for i in range(7, len(yp) + 1):
        su = 0
        sd = 0
        for j in yp[i - 7:i]:
            if j >= 0:
                su = su + j
            elif j <= 0:
                sd = sd + abs(j)

        cmo[date[i]] = (su - sd) * 100 / (su + sd)
    return cmo


def HMA(dl):
    hma = {}
    date = dl[..., 1]
    hp = list(dl[..., 3])
    for i in range(7, len(hp)):
        nhp = 0
        for j in hp[i - 7: i]:
            nhp = nhp + j
        hma[date[i]] = nhp / 7
    return hma


def SMA(dl):
    sma = {}
    date = dl[..., 1]
    cp = dl[..., 5]
    for i in range(7, len(cp)):
        ncp = 0
        for j in cp[i - 7: i]:
            ncp = ncp + j
        sma[date[i]] = ncp / 7
    return sma


def WMA(dl):
    wma = {}
    date = dl[..., 1]
    cp = dl[..., 5]
    for i in range(7, len(cp)):
        j = cp[i - 7: i]
        z = (7 * j[-1] + 6 * j[-2] + 5 * j[-3] + 4 * j[-4] + 3 * j[-5] + 2 * j[-6] + 1 * j[
            -7]) / 7 + 6 + 5 + 4 + 3 + 2 + 1
        wma[date[i]] = z
    return wma


def ROC(dl):
    roc = {}
    date = dl[..., 1]
    cp = dl[..., 5]
    for i in range(7, len(cp)):
        ncp = cp[i - 7: i]
        roc[date[i]] = ncp[6] - ncp[0]
    return roc


def EMA12(dl, y_ema):
    ema = y_ema
    date = list(dl[..., 1])
    for i, j in enumerate(dl[..., 5]):
        ema[date[i]] = (ema[list(ema.keys())[-1]] * 11 / 13) + j * 2 / 13
    del ema[list(ema.keys())[0]]
    return ema


def EMA26(dl, y_ema):
    ema = y_ema
    date = list(dl[..., 1])
    for i, j in enumerate(dl[..., 5]):
        ema[date[i]] = (ema[list(ema.keys())[-1]] * 25 / 27) + j * 2 / 27
    del ema[list(ema.keys())[0]]
    return ema


# middle factor
def EMA15(dl, y_ema):
    ema = y_ema
    date = list(dl[..., 1])
    for i, j in enumerate(dl[..., 5]):
        ema[date[i]] = (ema[list(ema.keys())[-1]] * 14 / 16) + j * 2 / 16
    del ema[list(ema.keys())[0]]
    return ema


def DEMA(ema15):
    dema = {}
    date = list(ema15.keys())
    for i in range(1, len(date)):
        if dema == {}:
            y_ema = ema15[date[0]]
        else:
            y_ema = dema[list(dema.keys())[-1]]
        dema[date[i]] = ema15[date[i]] * 2 - (y_ema * 14 / 16 + ema15[date[i]] * 2 / 16)
    return dema


def TEMA(ema15, dema):
    tema = {}
    date = list(dema.keys())
    for i in range(1, len(date)):
        if tema == {}:
            y_tema = dema[date[0]]
        else:
            y_tema = tema[list(tema.keys())[-1]]
        tema[date[i]] = dema[date[i]] + ema15[date[i]] - (y_tema * 14 / 16 + dema[date[i]] * 2 / 16)
    return tema


def DIF(ema12, ema26):
    dif = {i: ema12[i] - ema26[i] for i in list(ema12.keys())}
    return dif


def DEA(dif):
    dea = {}
    ldif = list(dif.keys())
    for i in range(1, len(dif)):
        dea[ldif[i]] = dif[ldif[i]] * 8 / 10 + dif[ldif[i - 1]] * 2 / 10
    return dea


def MACD(dif, dea):
    macd = {}
    for i in list(dif.keys()):
        j, k = dif.get(i), dea.get(i)
        if j is not None and k is not None:
            macd[i] = (j - k) * 2
    return macd


def avg(lis):
    s = sum(lis) / len(lis)
    return s


# 计算7天近一日的ROC列表
def one_roc(roc):
    for i in range(0, len(roc)):
        if roc[i] == 'null' and i != 0:
            roc[i] = roc[i - 1]
        elif roc[i] == 'null' and i == 0:
            for j, k in enumerate(roc):
                if k != 'null':
                    roc[i] = k
                    break
                elif j == len(roc) - 1:
                    roc = [1 for _ in range(0, len(roc))]
        else:
            pass
    return [(roc[i] - roc[i - 1]) / roc[i - 1] for i in range(len(roc) - 1, 0, -1)]


def Rsd(rank_i_l, rank_j_l, rank_i_t_a, rank_j_t_a):
    i_a = rank_i_t_a
    j_a = rank_j_t_a
    mini = min(len(rank_i_l), len(rank_j_l))
    rank_i_l, rank_j_l = torch.from_numpy(rank_i_l[:mini]), torch.from_numpy(rank_j_l[:mini])
    alpha = torch.sum((torch.tensor(rank_i_l).cuda() - i_a) * (torch.tensor(rank_j_l).cuda() - j_a)) \
            / torch.sqrt(torch.sum((torch.tensor(rank_i_l).cuda() - i_a) ** 2)) * torch.sqrt(torch.sum((torch.tensor(rank_j_l).cuda() - j_a) ** 2))
    print(alpha)
    return float(alpha)


def rt(i_data):  # i.data = [[i.date,i.cp],[],...[]]
    i_data.sort(key=lambda x: x[0])
    rit = {i_data[i][0]: math.log(i_data[i][1]) - math.log(i_data[i - 1][1]) for i in range(1, len(i_data))}
    return rit


def rt_mean(rt_list):
    rt_list.sort(key=lambda x: x[0])
    rt_list = np.array(rt_list)
    mean = {rt_list[i][0]: np.mean(rt_list[:i + 1, 1]) for i in range(0, len(rt_list))}
    return mean
