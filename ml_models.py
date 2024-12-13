# ml_models.py
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, GRU, Dropout, Dense, Attention, GlobalAveragePooling1D
from keras.models import Model
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler


class Gru_HL_Mean_Model:
    
    def __init__(self):
        self.model_id = 0
        self.st_name = 'None'
        self.trade_size = 0
        self.target_trade_profit = 0
        self.trade_loss_limit = 0
        self.test_end_date = 0
        self.max_trade_duration = 0
        self.training_duration = 0
        self.test_duration = 0

        self.screened_df = 0

    # Utility functions
    def create_sequences_multifeature(self, data, target_feature, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])  # Collect the window of all features
            y.append(data[i + window_size, target_feature])  # Predict only the target feature (e.g., "high")
        return np.array(X), np.array(y)


    def create_avg_seq(self, data, target_feature, trade_window):
        act_avg_price_seq = []
        for i in range(len(data) - trade_window):
            row = []
            for j in range(trade_window):
                row.append(data[i+j+1 , target_feature])  
            act_avg_price_seq.append(row)
        return np.array(act_avg_price_seq)


    def js_r(self, filename:str):
        with open(filename) as f_in:
            return json.load(f_in)

    def plot_to_base64(self, fig):
        """Convert a Matplotlib figure to a Base64 encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


    def check_model(self, model_id: int, stock_name: str, trade_size: float, target_trade_profit: float,
                     trade_loss_limit: float, test_end_date: str, max_trade_duration: int,
                     training_duration: int, test_duration: int):
        self.model_id = model_id
        self.st_name = stock_name 
        self.trade_size = trade_size
        self.target_trade_profit = target_trade_profit
        self.trade_loss_limit = trade_loss_limit
        self.test_end_date = test_end_date
        self.max_trade_duration = max_trade_duration
        self.training_duration = training_duration
        self.test_duration = test_duration
        # read data file into a df from the screened data directory
        # Variable to store the directory name
        self.direc_name = 'models' + '/' + str(model_id)
        self.screened_df = pd.read_pickle(f"{self.direc_name}/screening/Screened_{ self.st_name}.pkl")
        #print(self.screened_df.tail(5))




        # Check that we have or can get a valid test_end_date
        preds, actual, loss  = self.process_to_full_df() 
 
        return preds, actual, loss   




    def process_to_full_df(self) :   
        
        # Create new features
        # GEt column with the average of the High & Low prices
        self.screened_df['average'] = self.screened_df[['high','low']].mean(axis=1)
        # Get column with the daily percentage change of the average price
        self.screened_df['pct_change_avg'] = self.screened_df['average'].pct_change()
        
        df2 = self.screened_df.iloc[100:-self.max_trade_duration]

        features = df2[[ 'average', 'pct_change_avg']].values

        #print(features.tail(5))
        target_feature_index = 0  # Index of "high" in the features array


        # Scaler for input features (average & pct_change_avg)
        feature_scaler = MinMaxScaler()
        features_normalized = feature_scaler.fit_transform(features)


        # Scaler for target feature (average)
        target_scaler = MinMaxScaler()
        high_normalized = target_scaler.fit_transform(features[:, target_feature_index].reshape(-1, 1))  # Only scale "average"

        # Create sequences
        window_size = 25

        ml_settings = {"window_size" : window_size}
        #Store window size for later
        # Variable to store the directory name
        directory_name = self.direc_name + '/initial_training'
        # Create the directory if it doesn't exist
        os.makedirs(directory_name, exist_ok=True)
        # Save dictionarys to jason file
        with open(f"{directory_name}/ml_settings.json", "w") as final:
            json.dump(ml_settings, final)

        X, y = self.create_sequences_multifeature(features_normalized, target_feature_index, window_size)

        # Adjust X shape for the GRU model (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Shape (num_samples, window_size, num_features)

        train_days = self.training_duration - window_size

        X_train, X_test = X[:train_days], X[train_days:]
        y_train, y_test = y[:train_days], y[train_days:]

        # Input Layer
        input_layer = Input(shape=(X.shape[1], X.shape[2]))

        # GRU Layers with Dropout
        gru_output = GRU(50, return_sequences=True)(input_layer)
        gru_output = Dropout(0.2)(gru_output)

        gru_output = GRU(50, return_sequences=True)(gru_output)
        gru_output = Dropout(0.2)(gru_output)

        gru_output = GRU(50, return_sequences=True)(gru_output)
        gru_output = Dropout(0.2)(gru_output)

        gru_output = GRU(50, return_sequences=True)(gru_output)
        gru_output = Dropout(0.2)(gru_output)

        # Attention Layer
        attention_output = Attention()([gru_output, gru_output])

        # Global Average Pooling Layer
        pooled_output = GlobalAveragePooling1D()(attention_output)

        # Fully Connected Output Layer
        output_layer = Dense(1)(pooled_output)

        # Define Model
        attention_model = Model(inputs=input_layer, outputs=output_layer)
        attention_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        history = attention_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=80,
                batch_size=16,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )

        # Predict
        predicted_gru = attention_model.predict(X_test)
        loss = history.history

        # Inverse transform the predicted and actual values
        predicted_gru_actual = target_scaler.inverse_transform(predicted_gru)  # Use target scaler for "high"
        y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape y_test to (num_samples, 1)
       
        model_path = f"{self.direc_name}/initial_training/model.keras"
        attention_model.save(model_path)

        feature_scaler_path = f"{self.direc_name}/initial_training/feature_scalar.joblib"
        target_scaler_path = f"{self.direc_name}/initial_training/target_scalar.joblib"

        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)

        print("finished Running Model")

        return predicted_gru_actual,  y_test_actual, loss



    def backtest(self, model_id: int, stock_name: str, trade_size: float, target_trade_profit: float,
                     trade_loss_limit: float, test_end_date: str, max_trade_duration: int,
                     training_duration: int, test_duration: int):
        self.model_id = model_id
        self.st_name = stock_name 
        self.trade_size = trade_size
        self.target_trade_profit = target_trade_profit
        self.trade_loss_limit = trade_loss_limit
        self.test_end_date = test_end_date
        self.max_trade_duration = max_trade_duration
        self.training_duration = training_duration
        self.test_duration = test_duration

        # window_size is read from a json file below
        self.window_size = 0


        # read data file into a df from the screened data directory
        # Variable to store the directory name
        self.direc_name = 'models' + '/' + str(model_id)
        self.screened_df = pd.read_pickle(f"{self.direc_name}/screening/Screened_{ self.st_name}.pkl")
        #print(self.screened_df.tail(5))

        self.screened_df['average'] = self.screened_df[['high','low']].mean(axis=1)
        self.screened_df['pct_change_avg'] = self.screened_df['average'].pct_change()

        #get window_size
        ml_info = self.js_r(f"{self.direc_name}/initial_training/ml_settings.json")
        self.window_size = ml_info["window_size"]
        # First task is to get a dataset with column of labels for test days
        df_labels = self.screened_df.iloc[100+self.training_duration:-self.max_trade_duration]
        
        #create a longer dataset to use for sequence creation
        df_avg_seq_in = self.screened_df.iloc[100+self.training_duration:]

        avg_seq_in = df_avg_seq_in.values
        avg_seq = self.create_avg_seq(avg_seq_in, 6, self.max_trade_duration)

        df_avg_seq_processed = pd.DataFrame(avg_seq)
        #print(df_avg_seq_processed.tail())

        # Now calculate Avg_labels
        df_avg_labels = self.screened_df.iloc[100+self.training_duration:-self.max_trade_duration]
        # Create a copy of the DataFrame to avoid modifying the original slice
        df_avg_labels = df_avg_labels.copy()

        # Add the 'date' column
        df_avg_labels['date'] = df_avg_labels.index

        # Re-index the trimmed DataFrame to ensure indices are continuous
        df_avg_labels = df_avg_labels.reset_index(drop=True)


        #print(df_avg_labels.tail())

        #calculate thresholds and add as columns
        df_avg_labels['high_threshold'] = ((self.target_trade_profit / self.trade_size) + 1) * df_avg_labels['average']
        df_avg_labels['low_threshold'] = df_avg_labels['average'] - ((self.trade_loss_limit / self.trade_size) * df_avg_labels['average'])
        #Find max and min of sequence columns - add as columns
        df_avg_labels['max_val'] = df_avg_seq_processed.iloc[:,1:].max(axis=1)
        df_avg_labels['min_val'] = df_avg_seq_processed.iloc[:,1:].min(axis=1)
        #Do logic check to see if hit threshold high and not low - add avg_label column
        df_avg_labels['avg_lab'] = np.where(
                (df_avg_labels['max_val'] > df_avg_labels['high_threshold']) & 
                (df_avg_labels['min_val'] > df_avg_labels['low_threshold']), 
                1, 
                0
        )


        print('Dataframe index details: ', df_avg_labels.index)
        print('Number of screened labels: ', len(df_avg_labels['labels']))
        print('Number of averaged labels: ', len(df_avg_labels['avg_lab']))
        print('Null label values: ', df_avg_labels['labels'].isnull().sum())
        print('Null averaged label values: ',df_avg_labels['avg_lab'].isnull().sum())

        # Count rows where both columns have the value 1
        count_two_ones = df_avg_labels[(df_avg_labels['labels'] == 1) & (df_avg_labels['avg_lab'] == 1)].shape[0]
        count_Label_one_avg_zero = df_avg_labels[(df_avg_labels['labels'] == 1) & (df_avg_labels['avg_lab'] == 0)].shape[0]
        count_Label_zero_avg_one = df_avg_labels[(df_avg_labels['labels'] == 0) & (df_avg_labels['avg_lab'] == 1)].shape[0]
        count_two_zero = df_avg_labels[(df_avg_labels['labels'] == 0) & (df_avg_labels['avg_lab'] == 0)].shape[0]
        print(f"Number of rows where both columns have 1: {count_two_ones}")
        print(f"Number of rows where false neg (label is true but avg is 0: {count_Label_one_avg_zero}")
        print(f"Number of rows where false pos: {count_Label_zero_avg_one}")
        print(f"Number of rows where both columns have 0: {count_two_zero}")

        plots = {}  # Ensure 'plots' dictionary is initialized
        # Data
        categories = [
                'Both 1',
                'False POS (Lab 0, Avg_Lab 1)',
                'False NEG (Lab 1, Avg_Lab 0)',
                'Both 0'
        ]
        counts = [
                count_two_ones, 
                count_Label_zero_avg_one, 
                count_Label_one_avg_zero,
                count_two_zero
        ]

        # Bar chart
        fig = plt.figure(figsize=(11, 8))  # Assign figure to 'fig'
        # Create the bar chart
        plt.bar(categories, counts, color=['green', 'red', 'orange', 'blue'])
        # Add numeric values above the bars
        for i, count in enumerate(counts):
                plt.text(i, count + 1, str(count), ha='center', va='bottom')  # Adjust the `count + 1` for spacing

        # Add labels and title
        plt.title('Comparison of Label and Avg_Label Columns')
        plt.ylabel('Count')
        plt.xlabel('Categories')
        plt.xticks(rotation=45)  # Rotate category labels if needed
        plt.tight_layout()  # Adjust layout to avoid overlap
        plots["avg_label_plot"] = self.plot_to_base64(fig)  # Pass 'fig' instead of 'figure'
        plt.close(fig)  # Use 'fig' to close the figureplt.show()





        df2 = self.screened_df.iloc[100+self.training_duration-self.window_size :-self.max_trade_duration]


        print("f2 shape: ", df2.shape)


        features = df2[[ 'average', 'pct_change_avg']].values
        target_feature_index = 0  # Index of "high" in the features array

        feature_scaler_path = f"{self.direc_name}/initial_training/feature_scalar.joblib"
        target_scaler_path = f"{self.direc_name}/initial_training/target_scalar.joblib"

        feature_scaler = joblib.load(feature_scaler_path)
        features_normalized = feature_scaler.transform(features)


        target_scaler = joblib.load(target_scaler_path)
        avg_normalized = target_scaler.transform(features[:, target_feature_index].reshape(-1, 1))  # Only scale "high"

        # Create sequences
        #window_size = 25
        X, y = self.create_sequences_multifeature(features_normalized, target_feature_index, self.window_size)

        # Adjust X shape for the GRU model (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Shape (num_samples, window_size, num_features)

        # Iterate through each record
        for i in range(X.shape[0]):  # Loop over the number of samples
            # Extract the i-th record
            single_record = X[i]  # Shape: (25, 2)

            # Reshape to (1, 25, 2) for the GRU model
            single_record_reshaped = single_record.reshape(1, 25, 2)
                                
        # load up model from file
        paths_to_weights = f"{self.direc_name}/initial_training/model.keras"
        # load up model from file
        attention_model = keras.saving.load_model(paths_to_weights)

        pred_avg_price_seq = []
        # Iterate through each record
        for i in range(X.shape[0]):  # Loop over the number of samples
            # Extract the i-th record
            single_record = X[i]  # Shape: (25, 2)
            #print(single_record)
            # Reshape to (1, 25, 2) for the GRU model
            single_record_reshaped = single_record.reshape(1, 25, 2)
            #print(single_record)
            # List to store predictions
            future_predictions = []

            # Predict the next 5 days
            for _ in range(5):  # Adjust the range for the desired number of days
                # Predict the next value for the target feature
                predicted_value = attention_model.predict(single_record_reshaped, verbose=0)  # Shape: (1, 1)
                #print(predicted_value)

                single_pred_value = predicted_value[0, 0]
                #print(single_pred_value)
                predicted_value_scaled_back = target_scaler.inverse_transform(np.array(single_pred_value).reshape(-1, 1))
                #print(predicted_value_scaled_back)

                # Create a new feature vector for the predicted day
                new_day = single_record_reshaped[0, -1, :].copy()  # Start with the last timestep
                #print(new_day)

                #Scale it back to get values
                new_day_scaled_back = feature_scaler.inverse_transform(np.array(new_day).reshape(1, -1))
                #print(new_day_scaled_back)


                #Now calulcate the percent_change = pred price - last price / last price
                percent_change = ((predicted_value_scaled_back - new_day_scaled_back[0,0])/ new_day_scaled_back[0,0])
                #print(percent_change)


                # Assuming predicted_value_scaled_back and percent_change are scalars
                updated_line = np.array([predicted_value_scaled_back[0, 0], percent_change[0, 0]])
                #print("updated line: ", updated_line)

                # Update the appropriate columns in new_day
                target_index = target_feature_index  # Index for the target feature
                percent_change_index = target_feature_index + 1  # Assume next column for percent change

                # Ensure new_day is modified correctly
                new_day[target_index] = updated_line[0]  # Update with predicted value
                new_day[percent_change_index] = updated_line[1]  # Update with percent change

                #print("Updated new_day: ", new_day)

                # Rescale new_day
                new_day_rescaled = feature_scaler.transform(np.array(new_day).reshape(1, -1))  # Rescale to (1, num_features)
                new_day_rescaled = new_day_rescaled.flatten()  # Convert to 1D array for consistency
                #print("Updated & rescaled new_day: ", new_day_rescaled)

                #new_day_scaled_back = target_scaler.inverse_transform(np.array(new_day_num).reshape(-1, 1))
                #print(new_day_scaled_back)

                # Update the input sequence
                single_record_reshaped = np.append(single_record_reshaped[0, 1:, :], [new_day_rescaled], axis=0)  # Shape: (window_size, num_features)
                #print(single_record_reshaped)
                single_record_reshaped = single_record_reshaped.reshape((1, single_record_reshaped.shape[0], single_record_reshaped.shape[1]))
                #print(input_sequence)
                # Store the predicted value (scaled back if needed)
                future_predictions.append(predicted_value[0, 0])
                #break
        

            # Inverse transform the predictions (if data was scaled)
            future_predictions_scaled = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            pred_avg_price_seq.append(future_predictions_scaled)
            #print("Future Predictions (scaled back):", future_predictions_scaled)
            #break


        pred_seq= np.array(pred_avg_price_seq)
        # Reshape to 2D array of shape (200, 5)
        pred_seq_2d = pred_seq.reshape(pred_seq.shape[0], pred_seq.shape[1])
        print(pred_seq_2d.shape)  # Should output (200, 5)

        # Convert to pandas DataFrame
        df_pred_avg_seq_processed = pd.DataFrame(pred_seq_2d)

        #df_pred_avg_seq_processed = df_pred_avg_seq_processed.copy()
        #df_pred_avg_seq_processed = df_pred_avg_seq_processed.reset_index(drop=True)

        #print("processed DF: ", df_pred_avg_seq_processed.tail())
        #print("df_avg_labels: " , df_avg_labels.tail())
        #Find max and min of sequence columns - add as columns
        
        #df_avg_labels = pd.concat([df_avg_labels, df_pred_avg_seq_processed], axis =1)
        #print("df_avg_labels: " , df_avg_labels.shape)
        #df_avg_labels = df_avg_labels.copy()
        # Re-index the trimmed DataFrame to ensure indices are continuous
        #df_avg_labels = df_avg_labels.reset_index(drop=True)

        
        df_avg_labels['p_max_val'] = df_pred_avg_seq_processed.iloc[:,1:].max(axis=1)
        df_avg_labels['p_min_val'] = df_pred_avg_seq_processed.iloc[:,1:].min(axis=1)

        #Do logic check to see if hit threshold high and not low - add avg_label column
        df_avg_labels['pred_avg_lab'] = np.where(
                (df_avg_labels['p_max_val'] > df_avg_labels['high_threshold']) & 
                (df_avg_labels['p_min_val'] > df_avg_labels['low_threshold']), 
                1, 
                0
        )

        print('Dataframe index details: ', df_avg_labels.index)
        print('Number of screened labels: ', len(df_avg_labels['labels']))
        print('Number of predicted averaged labels: ', len(df_avg_labels['pred_avg_lab']))
        print('Null label values: ', df_avg_labels['labels'].isnull().sum())
        print('Null predicted averaged label values: ',df_avg_labels['pred_avg_lab'].isnull().sum())

        # Count rows where both columns have the value 1
        count_two_ones = df_avg_labels[(df_avg_labels['labels'] == 1) & (df_avg_labels['pred_avg_lab'] == 1)].shape[0]
        count_Label_one_p_avg_zero = df_avg_labels[(df_avg_labels['labels'] == 1) & (df_avg_labels['pred_avg_lab'] == 0)].shape[0]
        count_Label_zero_p_avg_one = df_avg_labels[(df_avg_labels['labels'] == 0) & (df_avg_labels['pred_avg_lab'] == 1)].shape[0]
        count_two_zero = df_avg_labels[(df_avg_labels['labels'] == 0) & (df_avg_labels['pred_avg_lab'] == 0)].shape[0]
        print(f"Number of rows where both columns have 1: {count_two_ones}")
        print(f"Number of rows where false neg (label is true but avg is 0: {count_Label_one_avg_zero}")
        print(f"Number of rows where false pos: {count_Label_zero_avg_one}")
        print(f"Number of rows where both columns have 0: {count_two_zero}")

        # Data
        categories = [
                'Both 1',
                'False POS (Lab 0, PRED_Avg_Lab 1)',
                'False NEG (Lab 1, PRED_Avg_Lab 0)',
                'Both 0'
        ]
        counts = [
                count_two_ones, 
                count_Label_zero_p_avg_one, 
                count_Label_one_p_avg_zero,
                count_two_zero
        ]

        # Bar chart
        fig = plt.figure(figsize=(11, 8))  # Assign figure to 'fig'
        # Create the bar chart
        plt.bar(categories, counts, color=['green', 'red', 'orange', 'blue'])

        # Add numeric values above the bars
        for i, count in enumerate(counts):
                plt.text(i, count + 1, str(count), ha='center', va='bottom')  # Adjust the `count + 1` for spacing

        # Add labels and title
        plt.title('Comparison of Label and PREDICTED Avg_Label Columns')
        plt.ylabel('Count')
        plt.xlabel('Categories')
        plt.xticks(rotation=45)  # Rotate category labels if needed
        plt.tight_layout()  # Adjust layout to avoid overlap
        plots["pred_avg_label_plot"] = self.plot_to_base64(fig)  # Pass 'fig' instead of 'figure'
        plt.close(fig)  # Use 'fig' to close the figureplt.show()
        

        # Variable to store the directory name
        directory_name = self.direc_name + '/initial_training'
        # Create the directory if it doesn't exist
        os.makedirs(directory_name, exist_ok=True)
        # Save dictionarys to jason file
        with open(f"{directory_name}/label_plots.json", "w") as final:
            json.dump(plots, final)







        # Check that we have or can get a valid test_end_date
        #preds, actual, loss  = self.process_to_full_df() 
 
        return #preds, actual, loss



