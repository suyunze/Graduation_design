import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import registry

from databaseORM import db, item
from factors import *


class factors_cul(object):
    def __init__(self, target_stock):
        self.target_stock = target_stock
        self.metadata_obj = MetaData()
        mapper_registry = registry()
        self.Base = mapper_registry.generate_base()
        self.engine, self.session = db().content()
        self.Item = item(target_stock)
        data = self.session.query(self.Item).order_by(self.Item.date)
        train_data = []
        for i in data:
            train_data.append([i.code, i.date, i.bp, i.hp, i.lp, i.sp, i.yp, i.udp, i.udr])
        self.al = np.array(train_data)

    def cul_factors(self):
        ema12 = EMA12(dl=self.al[1:, ...], y_ema={self.al[0][1]: self.al[0][5]})
        ema26 = EMA26(dl=self.al[1:, ...], y_ema={self.al[0][1]: self.al[0][5]})
        ema15 = EMA15(dl=self.al[1:, ...], y_ema={self.al[0][1]: self.al[0][5]})
        dema = DEMA(ema15)
        dif = DIF(ema12, ema26)
        dea = DEA(dif)

        cci = CCI(self.al)
        macd = MACD(dif, dea)
        wr = WR(self.al)
        cmo = CMO(self.al)
        ema = ema12
        hma = HMA(self.al)
        sma = SMA(self.al)
        wma = WMA(self.al)
        tema = TEMA(ema15, dema)
        roc = ROC(self.al)
        return cci, macd, wr, cmo, ema, hma, sma, wma, tema, roc

    def update_factors(self):
        cci, macd, wr, cmo, ema, hma, sma, wma, tema, roc = self.cul_factors()
        cci = {k: (cci[k] - min(cci.values())) / (max(cci.values()) - min(cci.values())) for k in cci.keys()}
        macd = {k: (macd[k] - min(macd.values())) / (max(macd.values()) - min(macd.values())) for k in macd.keys()}
        wr = {k: (wr[k] - min(wr.values())) / (max(wr.values()) - min(wr.values())) for k in wr.keys()}
        cmo = {k: (cmo[k] - min(cmo.values())) / (max(cmo.values()) - min(cmo.values())) for k in cmo.keys()}
        ema = {k: (ema[k] - min(ema.values())) / (max(ema.values()) - min(ema.values())) for k in ema.keys()}
        hma = {k: (hma[k] - min(hma.values())) / (max(hma.values()) - min(hma.values())) for k in hma.keys()}
        sma = {k: (sma[k] - min(sma.values())) / (max(sma.values()) - min(sma.values())) for k in sma.keys()}
        wma = {k: (wma[k] - min(wma.values())) / (max(wma.values()) - min(wma.values())) for k in wma.keys()}
        tema = {k: (tema[k] - min(tema.values())) / (max(tema.values()) - min(tema.values())) for k in tema.keys()}
        roc = {k: (roc[k] - min(roc.values())) / (max(roc.values()) - min(roc.values())) for k in roc.keys()}

        mapping = {i: {'CCI': cci.get(i, null), 'MACD': macd.get(i, null), 'WR': wr.get(i, null),
                       'CMO': cmo.get(i, null), 'EMA': ema.get(i, null), 'HMA': hma.get(i, null),
                       'SMA': sma.get(i, null),
                       'WMA': wma.get(i, null), 'TEMA': tema.get(i, null), 'ROC': roc.get(i, null)} for i in cci.keys()}
        obj = self.session.query(self.Item).all()
        for i in obj:
            try:
                i.cci = mapping[i.date]['CCI']
                i.macd = mapping[i.date]['MACD']
                i.wr = mapping[i.date]['WR']
                i.cmo = mapping[i.date]['CMO']
                i.ema = mapping[i.date]['EMA']
                i.hma = mapping[i.date]['HMA']
                i.sma = mapping[i.date]['SMA']
                i.wma = mapping[i.date]['WMA']
                i.tema = mapping[i.date]['TEMA']
                i.roc = mapping[i.date]['ROC']
            except KeyError:
                pass
        self.session.flush()
        self.session.commit()



# factors_cul('000001.sz').update_factors()
