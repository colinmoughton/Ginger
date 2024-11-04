from sqlalchemy import Boolean, Column, Integer, String, Float, Date, Table, MetaData
from sqlalchemy.orm import registry
from database import Base, engine

# Metadata registry for dynamic table creation
metadata_obj = MetaData()

class PredictiveModel(Base):
    __tablename__ = "PredictiveModels"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    complete = Column(Boolean, default=False)



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
        metadata_obj,
        Column("id", Integer, primary_key=True),
        Column("date", Date),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Integer),
    )    
    metadata_obj.create_all(engine)  # Create the table in the database
    return stock_table
