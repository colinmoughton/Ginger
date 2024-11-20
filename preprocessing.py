# preprocessing.py
import pandas as pd
import os

class PreProcessing:
    
    def __init__(self):
        self.st_name = 'None'
        self.direc_name = 'None'
        self.index_loc = 0
        self.rec_before = 0
        self.rec_after = 0
        self.trim_df = 0
        self.hits_dict = 0
        self.labeled_df = 0
        self.metrics_dict = {}

    def screen_stock(self, model_id: int, stock_name: str, stock_table: pd.DataFrame, trade_size: float, target_trade_profit: float,
                     trade_loss_limit: float, test_end_date: str, max_trade_duration: int,
                     training_duration: int, test_duration: int, sma_1_duration: int,
                     sma_2_duration: int, sma_3_duration: int, gen_files: bool):
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
        self.st_name = stock_name 
        
        # Reverse the dataframe index order earliest to latest
        stock_table.sort_index(inplace=True)
        #print(stock_table.head())

        # Check that we have or can get a valid test_end_date
        if self.check_target_date(stock_table, test_end_date) == False:
            return "Invalid test end date"

        # Check that we have or can get a valid test_end_date
        if self.check_enough_data(stock_table, max_trade_duration,
                     training_duration, test_duration, sma_1_duration,
                     sma_2_duration, sma_3_duration) == False:
            return "Not enough data for this stock"

        if self.slice_and_label(model_id, stock_name, stock_table, trade_size, target_trade_profit, 
                                trade_loss_limit, max_trade_duration) == False:
            return "Stuck in slicing"

        if self.get_metrics(gen_files) == False:
            return "Stuck in slicing"
        else:
            return self.metrics_dict






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
        self.rec_before = records_before
        #print("Records Before: " , records_before)

        records_after = max_trade_duration
        self.rec_after = records_after
        #print("Records After: " , records_after)

        # Check there are enough records before and after
        has_records_before = self.index_loc >= records_before
        #print("Index_loc: " , self.index_loc)
        #print("Has records before: " , has_records_before)

        has_records_after = self.index_loc + records_after < len(stock_table)
        #print("Has records after: " , has_records_after)

        if has_records_before and has_records_after:
            return True
        else:
            return False


    def slice_and_label(self, model_id: int, stock_name: str, stock_table: pd.DataFrame, 
                        trade_size: float, target_trade_profit: float,
                     trade_loss_limit: float, max_trade_duration: int ):   
        # Calculate the start and end positions for slicing
        start_pos = self.index_loc - self.rec_before
        end_pos =  self.index_loc + self.rec_after + 1  # +1 to include the row at `index_location + records_after`

        # Slice the DataFrame
        trimmed_df = stock_table.iloc[start_pos:end_pos]

        #self.trim_df=trimmed_df
        # Print the result or inspect
        #print("Trimmed DataFrame:")
        #print(trimmed_df.head())
        #print(trimmed_df.tail())
        #df =trimmed_df.copy()

        # Copy the date index values into a new column called 'date'
        trimmed_df['date'] = trimmed_df.index  #.copy()

        # Re-index the trimmed DataFrame to ensure indices are continuous
        trimmed_df = trimmed_df.reset_index(drop=True)

        # Initialize the 'labels' column in the trimmed DataFrame with 0 as the default value
        trimmed_df['labels'] = 0
        # To inspect the DataFrame with the new column
        #print(trimmed_df.head())
        # Prepare an empty list to capture the number of days it took for each record to meet the first condition
        days_to_hit_high_threshold_dict = {}
        
        # Loop through each row until reaching `len(trimmed_df) - 15` to stay within bounds
        for i in range(len(trimmed_df) - max_trade_duration):

            # Define the start record as the current row
            start_value = (trimmed_df.iloc[i]["high"] + trimmed_df.iloc[i]["low"]) / 2
            
            # Set thresholds for high and low based on start_value
            high_threshold = ((target_trade_profit / trade_size) + 1) * start_value 
            low_threshold = start_value - ((trade_loss_limit / trade_size) * start_value)

            #print('High Threshold: ', high_threshold)
            #print('Low Threshold: ', low_threshold)

            # Initialize flags for conditions
            high_condition_met = False
            low_condition_met = False

            # Check the next 15 records for each condition
            for j in range(1, max_trade_duration):

                # Ensure `i + j` does not exceed the DataFrame length
                #if i + j >= len(trimmed_df):
                #    break

                # Access the row j days after the current record
                current_high = trimmed_df.iloc[i + j]['high']
                current_low = trimmed_df.iloc[i + j]['low']

                # Check if high threshold is met and capture the days taken to meet it
                if not high_condition_met and current_high > high_threshold:
                    high_condition_met = True
                    days_to_hit_high_threshold_dict[i] = j  # Save the days it took to reach the threshold
                    break
                # Check if low threshold is met
                if not low_condition_met and current_low < low_threshold:
                    low_condition_met = True
                    break
            

            # If both conditions are met, label the current record with a 1
            if high_condition_met:
                trimmed_df.at[i, 'labels'] = 1
                


        # Print the resulting DataFrame and the days to hit the high threshold
        #print(trimmed_df.head(100))
        #print("Days to hit high threshold for each record:", days_to_hit_high_threshold_dict)
        #trimmed_df.to_csv('SN-Trim.csv') 
        self.hits_dict = days_to_hit_high_threshold_dict
        self.labeled_df = trimmed_df

        # Variable to store the directory name
        directory_name = 'models' + '/' + str(model_id) + '/screening' 
        
        self.direc_name = directory_name
        # Create the directory if it doesn't exist
        os.makedirs(directory_name, exist_ok=True)
        #print(f"Directory '{directory_name}' is ready (created if it didn't exist).")
        #trimmed_df.to_pickle(f"{directory_name}/Sliced{stock_name}.pkl")
        

        return True



    def get_metrics(self, gen_files: bool):   
        # Calculate the start and end positions for slicing

        df_hits = pd.DataFrame(self.hits_dict.items(), columns=['position' , 'time_to_top'])
        #print(df_hits.head())

        # Work out total records
        total_records1 = self.rec_before + 1 + self.rec_after
        total_records2 = len(self.labeled_df)

        # Work out occurance
        occurance = len(df_hits)

        if occurance ==0: return False

        # Work out Occurance Interval
        occ_interval = total_records1 / occurance

        # Work out ave_duration
        ave_duration = df_hits['time_to_top'].mean()

        # Work out 4 sigma std dev
        four_sigma = df_hits['time_to_top'].std() * 4

        #print("total_records1: ", total_records1)
        #print("total_records2: ", total_records2)

        #print("Occurance: ", occurance)
        #print("Occurance interval: ", occ_interval)

        #print("Average duration: ", ave_duration)

        #print("Four sigma: ", four_sigma)


        self.metrics_dict = {
            'stock_name' : self.st_name ,
            'total_records' : total_records1 , 
            'occurance' : occurance ,
            'occ_interval' : occ_interval ,
            'ave_duration' : ave_duration.item() ,
            'four_sigma' : four_sigma.item() }

        self.labeled_df['date'] = pd.to_datetime(self.labeled_df['date'])
        #print(self.labeled_df.head())
        #print(self.labeled_df.dtypes)

        self.labeled_df.index = self.labeled_df.date
        self.labeled_df = self.labeled_df.drop('date',axis=1)
        
        #print(self.labeled_df.head())
        #print(self.labeled_df.dtypes)
        if gen_files:
            self.labeled_df.to_pickle(f"{self.direc_name}/Screened_{ self.st_name}.pkl")
