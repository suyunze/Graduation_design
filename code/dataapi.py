# 导入tushare

import tushare as ts


class dataapi(object):
    def __init__(self):
        # 初始化pro接口
        self.pro = ts.pro_api('**********************************')

    def stock_basic(self):
        df = self.pro.stock_basic(**{
            "ts_code": "",
            "name": "",
            "exchange": "",
            "market": "",
            "is_hs": "",
            "list_status": "",
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "list_date",
            "fullname",
            "curr_type",
            "delist_date"
        ])
        return df

    def daily(self, tscode, start, end):
        df = self.pro.daily(**{
            "ts_code": tscode,
            "trade_date": "",
            "start_date": start,
            "end_date": end,
            "offset": "",
            "limit": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
        return df

print(dataapi().daily('000004.SZ', '20170101', '20171231'))
