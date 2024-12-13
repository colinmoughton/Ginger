from sqlalchemy import Boolean, Column, Integer, String, Float, Date
from sqlalchemy.orm import registry
from database import Base, engine

class PredictiveModel(Base):
    __tablename__ = "PredictiveModels"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    complete = Column(Boolean, default=False)

    # New columns for individual model details
    trade_size = Column(Float)
    target_trade_profit = Column(Float)
    trade_loss_limit = Column(Float)
    test_end_date = Column(Date)
    max_trade_duration = Column(Integer)
    training_duration = Column(Integer)
    test_duration = Column(Integer)
    sma_1_duration = Column(Integer)
    sma_2_duration = Column(Integer)
    sma_3_duration = Column(Integer)
    selected_stock = Column(String)
    total_records = Column(Integer)
    Occurance = Column(Integer)
    occ_interval = Column(Float)
    ave_duration = Column(Float)
    four_sigma = Column(Float)

class StockList(Base):
    __tablename__ = "stock_list"
            
    id = Column(Integer, primary_key=True, index=True)
    stock_name = Column(String, unique=True, index=True)
    start_date = Column(Date)
    end_date = Column(Date)
    number_records = Column(Integer)

def create_stock_table(stock_name):
    """Dynamically creates a stock data table based on stock_name."""
    stock_table = Table(
        stock_name,
        Base.metadata,
        Column("id", Integer, primary_key=True),
        Column("date", Date),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Integer),
    )    
    Base.metadata.create_all(engine)  # Create the table in the database
    return stock_table

