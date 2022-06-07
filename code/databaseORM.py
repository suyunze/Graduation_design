from sqlalchemy import *
from sqlalchemy.orm import *
from sqlalchemy.orm import registry

metadata_obj = MetaData()
mapper_registry = registry()
Base = mapper_registry.generate_base()


class db(object):
    def __init__(self):
        self.databaseDriver = 'mysql+pymysql://root:**********@localhost:3306/stock?charset=utf8'

    def content(self):
        try:
            eng = create_engine(self.databaseDriver, encoding='utf-8')
            Ses = sessionmaker(eng)
            ses = Ses()
            print('连接MySQL数据库成功')
        except Exception as e:
            print('连接数据库失败', e)

        return eng, ses


class Stock(Base):
    __tablename__ = 'stock_list'
    tscode = Column('tscode', String(9), primary_key=True)
    code = Column('code', String(6), primary_key=True)
    name = Column('name', String(256))
    area = Column('area', String(64), nullable=True)
    industry = Column('industry', String(256), nullable=True)
    fullname = Column('fullname', String(256))
    market = Column('market', String(16))
    type = Column('curr_type', String(4))
    listdate = Column('list_date', String(8))
    delist = Column('delist', String(8), nullable=True)

    def __repr__(self):
        return f"Stock(tscode={self.tscode!r}, code={self.code!r}, name={self.name!r}, area={self.area!r}, industry={self.industry!r}, fullname={self.fullname!r}, market={self.market!r}, type={self.type!r}, list_date={self.listdate!r}), delist={self.delist!r})"


class Predict(Base):
    __tablename__ = 'predict'
    code = Column('code', String(6), primary_key=True)
    date = Column('data', String(6))
    preview = Column('price', Float)

    def __repr__(self):
        return f"code={self.code!r},date={self.date!r},price={self.preview!r}"


def item(tablename):
    class Item(Base):
        __tablename__ = tablename
        __table_args__ = {'extend_existing': True}
        code = Column('code', String(9))  # 股票代码code
        date = Column('date', DATE, primary_key=True)  # 日期date
        bp = Column('Begin_Price', Float)  # 开盘价Begin_Price
        hp = Column('Highest_Price', Float)  # 最高价Highest_Price
        lp = Column('Lowest_Price', Float)  # 最低价Lowest_Price
        sp = Column('Closing_Price', Float)  # 收盘价Closing_Price
        yp = Column('Yesterday_Price', Float)  # 前收盘Yesterday_Price
        udp = Column('Up_Down_Price', Float)  # 涨跌额Up_Down_Price
        udr = Column('Up_Down_Range', Float)  # 涨跌幅Up_Down_Range
        dn = Column('Deal_Num', BIGINT)  # 成交量Deal_Num
        dp = Column('Deal_Price', BIGINT)  # 成交额Deal_Price
        cci = Column('CCI', Float)  # CCI
        macd = Column('MACD', Float)  # MACD
        wr = Column('WR', Float)  # WR
        cmo = Column('CMO', Float)  # CMO
        ema = Column('EMA', Float)  # EMA
        hma = Column('HMA', Float)  # HMA
        sma = Column('SMA', Float)  # SMA
        wma = Column('WMA', Float)  # WMA
        tema = Column('TEMA', Float)  # TEMA
        roc = Column('ROC', Float)  # ROC
        rt = Column('Rt', Float)  # Rank在t时刻的值
        rta = Column('Rta', Float)  # Rank在t时刻的均值

    return Item
