#  Copyright (c) 2022 by syz
import datetime
import json

from sqlalchemy.exc import IntegrityError, PendingRollbackError

from dataapi import dataapi
from databaseORM import *


def stock_list(tof):
    Base.metadata.create_all(engine, checkfirst=True)
    basicdata = dataapi().stock_basic()
    basicdata = basicdata.values
    for i in basicdata:
        x = Stock(tscode=i[0], code=i[1], name=i[2], area=i[3], industry=i[4], fullname=i[5], market=i[6], type=i[7],
                  listdate=i[8], delist=i[9])
        if tof is True:
            info[i[0]] = ''
        else:
            if i[0] not in info.keys():
                info[i[0]] = ''
            else:
                pass
        session.add(x)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
    session.commit()


def items(code, start=''):
    stock_item = item(code)
    eng, sess = db().content()
    Base.metadata.create_all(eng, checkfirst=True)
    dailydata = dataapi().daily(tscode=code, start=start)
    dailydata = dailydata.values
    date = datetime.datetime.now().strftime("%Y%m%d")
    for i in dailydata:
        x = stock_item(code=i[0], date=i[1], bp=i[2], hp=i[3], lp=i[4], sp=i[5], yp=i[6], udp=i[7], udr=i[8], dn=i[9],
                       dp=i[10])
        sess.add(x)
        try:
            sess.flush()
        except (PendingRollbackError, IntegrityError):
            sess.rollback()
    try:
        sess.commit()
    except IntegrityError:
        pass
    info[code] = date
    file.flush()
    print(code + '    完成')


def data():
    stocklist = session.query(Stock).all()  # .filter(Stock.code > 688728)
    for i in stocklist:
        code, start = i.tscode, info[i.tscode]
        items(code, start)
    file.write(json.dumps(info))
    file.flush()


engine, session = db().content()
with open("./info/list.txt", 'r') as t:
    f = t.read()
    t.close()
file = open("./info/list.txt", 'w+')
if f == "":
    print("首次使用，历史数据为空")
    info = {}
    tf = True
else:
    print("非首次使用，增加数据")
    info = json.loads(f)
    tf = False

stock_list(tof=tf)
data()
