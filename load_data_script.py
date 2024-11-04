import os
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import StockList, create_stock_table
from sqlalchemy import insert

# Define the data directory containing the .pkl files
DATA_DIR = "data"

def process_files():
    db = SessionLocal()
    try:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".pkl"):
                stock_name = filename.replace(".pkl", "")
                
                # Read the stock data from the .pkl file
                df = pd.read_pickle(os.path.join(DATA_DIR, filename))
                
                # Get metadata information
                start_date = df.index.min()
                end_date = df.index.max()
                number_records = len(df)
                
                # Add stock information to StockList table
                stock_info = StockList(
                    stock_name=stock_name,
                    start_date=start_date,
                    end_date=end_date,
                    number_records=number_records
                )
                db.add(stock_info)
                db.commit()  # Commit to get the generated ID

                # Create a unique table for each stock and populate it with data
                stock_table = create_stock_table(stock_name)

                # Insert stock data into the newly created table
                data_to_insert = [
                    {
                        "date": row.Index, 
                        "open": row.Open, 
                        "high": row.High, 
                        "low": row.Low, 
                        "close": row.Close, 
                        "volume": row.Volume
                    }
                    for row in df.itertuples()
                ]
                db.execute(insert(stock_table), data_to_insert)
                db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    process_files()
