import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score
import pickle
import warnings
warnings.simplefilter('ignore')

df_22 = pd.read_excel(r"C:\Users\Programming.com\Desktop\08_02_24\SalesG_2022.xlsx")
df_23 = pd.read_excel(r"C:\Users\Programming.com\Desktop\08_02_24\Sales.xlsx")
df_24 = pd.read_excel(r"C:\Users\Programming.com\Downloads\Sales Order.xlsx",sheet_name='Sheet2',skiprows=0)
df_24_mar = pd.read_excel(r"C:\Users\Programming.com\Downloads\s.xlsx")
df_24_mar = df_24_mar[['OrderDate','m2 sold']].copy()
# Step 1: Load time series data into DataFrame
def load_time_series_data(df_22,df_23,df_24,df_24_mar):
    df_concatenated = pd.concat([df_22[['InvoiceDate', 'm² Sold']], 
                             df_23[['InvoiceDate', 'm² Sold']],
                             df_24.rename(columns={'OrderDate': 'InvoiceDate','m2 sold' : 'm² Sold'})[['InvoiceDate', 'm² Sold']],
                             df_24_mar.rename(columns={'OrderDate': 'InvoiceDate','m2 sold' : 'm² Sold'})[['InvoiceDate', 'm² Sold']]],
                            ignore_index=True)

    df_concatenated.dropna(subset=['InvoiceDate'],inplace=True)
    df_concatenated.InvoiceDate = pd.to_datetime(df_concatenated.InvoiceDate)
    df_concatenated['Month'] = df_concatenated.InvoiceDate.dt.to_period('M')
    df_concatenated.Month = df_concatenated.Month.dt.to_timestamp()
    df_22_24 = df_concatenated[['Month','m² Sold']].copy()
    df_22_24.Month = pd.to_datetime(df_22_24.Month)

    return df_22_24

# Step 2: Preprocess time series data
def preprocess_time_series_data(df):
    # data preparation
    df.fillna(0,inplace=True)

    df = df[~(df['m² Sold']<0)]
    df_monthly = df[['Month','m² Sold']].groupby('Month').sum('m² Sold').reset_index()
    return df_monthly
###########################################################################################################
###########################################################################################################
p_values = [1, 2]  # Adjust as needed
d_values = [0,1]        # Adjust as needed
q_values = [1, 2]  # Adjust as needed


P_values = [0, 1]  # Adjust as needed
D_values = [0,1]        # Adjust as needed
Q_values = [0,1]  # Adjust as needed
s_values = [12]       # Seasonality, adjust based on your data
###########################################################################################################
hyperparameters = product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)


###########################################################################################################
###########################################################################################################


# Step 3: Train a machine learning model
def train_model(df,hyperparameters):

    # Initialize variables to store the best model and its performance
    best_rmse = float('inf')
    best_model = None
    best_order = None
    best_sessonal_order = None
    best_r2_score = None

    train_size = int(len(df)*0.8)
    train,test = df.iloc[:train_size,:],df.iloc[train_size:,:]



    try:
        for params in hyperparameters:
            p, d, q, P, D, Q, s = params
            order = (p,d,q)
            sessonal_order = (P,D,Q,s)
            
            # Fit SARIMAX model with current hyperparameters
            sarima_model = SARIMAX(df['m² Sold'], order=(p, d, q), seasonal_order=(P, D, Q, s))
            sarima_fit_model = sarima_model.fit(method='powell')

            # Predict on the testing set
            sarima_predictions = sarima_fit_model.get_prediction(start=test.index.min(), end=test.index.max(), dynamic=False)
            sarima_predictions_df = pd.DataFrame({'sarima_predictions_m2': sarima_predictions.predicted_mean.values}, index=test.index)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test['m² Sold'], sarima_predictions_df['sarima_predictions_m2']))
            r2 = r2_score(test['m² Sold'], sarima_predictions_df['sarima_predictions_m2'])
            # Update best model if current model has a lower RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = sarima_fit_model
                best_order = order
                best_sessonal_order = sessonal_order
                best_r2_score = r2
    except ValueError:
        print("Trying alternative initialization for SARIMA...")
    
    current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    with open(f"sarima{best_model.model.order + best_model.model.seasonal_order}_{current_date}.pkl",'wb')as file:
        pickle.dump(best_model, file)

    return best_model,best_rmse,best_r2_score,test

# Step 4: Evaluate the model
def evaluate_model(best_model,best_rmse, best_r2_score,test):

    print(f'Root_Mean Squared Error: {best_rmse}')
    print(f'Root_Mean Squared Error: {best_r2_score}')

    sarima_test = best_model.get_prediction(start=test.index.min(), end=test.index.max())

    sarima_test = sarima_test.predicted_mean
    sarima_test
    mse = mean_squared_error(test['m² Sold'],sarima_test)
    print(f'Mean Squared Error on Test Set: {mse}')
    print(f"RMSE: {np.sqrt(mse)}")
    Sarima_r2score = r2_score(test['m² Sold'],sarima_test)
    print(f'r2_score on Test Set: {Sarima_r2score}')
    SMAPE = mean_absolute_percentage_error(test['m² Sold'],sarima_test)
    print(f'MAPE on Test Set: {SMAPE}')

    return sarima_test




def forecast_future(best_model, evaluation, df_preprocessed, num_steps):
    print(best_model.summary())
    reports = df_preprocessed.copy()
    reports['sarima_test_predict'] = evaluation
    reports = reports.set_index('Month')
    
    forecat = best_model.get_forecast(steps=num_steps)
    forecat = forecat.predicted_mean.values
    # Create datetime index for forecasted values
    last_date = reports.index[-1]
    start_date = last_date + pd.DateOffset(months=1)  # Add one month to last date
    forecast_dates = pd.date_range(start=start_date, periods=num_steps, freq='MS')
    forecast_df = pd.DataFrame(forecat, index=forecast_dates, columns=['Sarima_Forecast'])
    reports = pd.concat([reports,forecast_df])
    
    return reports

    


# Step 5: Execute the pipeline
def main():
    # file_path = 'time_series_data.csv'  # Replace with your file path
    df = load_time_series_data(df_22,df_23,df_24,df_24_mar)
    print('Data ingetion_completed')
    df_preprocessed = preprocess_time_series_data(df)
    print('Data preprocessing completed')
    best_model,best_rmse,best_r2_score,test = train_model(df_preprocessed,hyperparameters)
    print('model training completed')
    evaluation = evaluate_model(best_model,best_rmse, best_r2_score,test)
    print('evalution completed')
    num_steps = 5
    report = forecast_future(best_model, evaluation, df_preprocessed, num_steps)
    report.to_excel("sarima_future_forecast.xlsx")
    print('report dumped')


if __name__ == "__main__":
    main()


