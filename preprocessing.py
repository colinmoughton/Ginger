# preprocessing.py
import pandas as pd

class PreProcessing:
    
    def __init__(self):
        self.index_loc = 0



    def screen_stock(self, stock_table: pd.DataFrame, trade_size: float, target_trade_profit: float,
                     trade_loss_limit: float, test_end_date: str, max_trade_duration: int,
                     training_duration: int, test_duration: int, sma_1_duration: int,
                     sma_2_duration: int, sma_3_duration: int):
        # Implement the screening logic here
        # Example: Perform calculations or filtering based on SMA durations and trade size
        # This function will return a result that indicates whether the stock meets the criteria

        # For demonstration, let's assume we compute some metric and return a boolean or score
        # Example dummy logic:
        # sma_1 = stock_table["close"].rolling(window=sma_1_duration).mean()
        # sma_2 = stock_table["close"].rolling(window=sma_2_duration).mean()
        # sma_3 = stock_table["close"].rolling(window=sma_3_duration).mean()

        # Simple condition: check if the last SMA values align with certain criteria
        #if sma_1.iloc[-1] > sma_2.iloc[-1] and sma_2.iloc[-1] > sma_3.iloc[-1]:
        #    return "Passed Screening"
        #else:
        #    return "Failed Screening"
        
        # Reverse the dataframe index order earliest to latest
        stock_table.sort_index(inplace=True)
        print(stock_table.head())

        # Check that we have or can get a valid test_end_date
        if self.check_target_date(stock_table, test_end_date) == False:
            return "Invalid test end date"

        # Check that we have or can get a valid test_end_date
        if self.check_enough_data(stock_table, max_trade_duration,
                     training_duration, test_duration, sma_1_duration,
                     sma_2_duration, sma_3_duration) == False:
            return "Not enough data for this stock"



    def check_target_date(self, stock_table: pd.DataFrame, test_end_date: str):   
        # Define the target date
        target_date = test_end_date

        # Check if the date exists within the index
        if target_date in stock_table.index:
            # Get the location
            index_location = stock_table.index.get_loc(target_date)
            print(f"Date {target_date} found at index location: {index_location}")
            self.index_loc = index_location
            return True
        else:
            # Find the closest prior date
            prior_dates = stock_table.index[stock_table.index < target_date]
            if prior_dates.empty:
                print("No previous date found in the dataset.")
                return False
            else:
                # Get the closest prior date and its index location
                closest_date = prior_dates.max()
                index_location = stock_table.index.get_loc(closest_date)
                self.index_loc = index_location
                print(f"Date {target_date} not found. Closest prior date is {closest_date} at index location: {index_location}")
                return True



    def check_enough_data(self, stock_table: pd.DataFrame, max_trade_duration: int,
                     training_duration: int, test_duration: int, sma_1_duration: int,
                          sma_2_duration: int, sma_3_duration: int):   
        # Define how many records needed before and after test_end_date
        
        # Records before - need to work out max SMA length first
        SMA_trail_length = max(sma_1_duration , sma_2_duration , sma_3_duration)

        records_before = SMA_trail_length + training_duration + test_duration + max_trade_duration
        print("Records Before: " , records_before)

        records_after = max_trade_duration
        print("Records After: " , records_after)

        # Check there are enough records before and after
        has_records_before = self.index_loc >= records_before
        print("Index_loc: " , self.index_loc)
        print("Has records before: " , has_records_before)

        has_records_after = self.index_loc + records_after < len(stock_table)
        print("Has records after: " , has_records_after)

        if has_records_before and has_records_after:
            return True

