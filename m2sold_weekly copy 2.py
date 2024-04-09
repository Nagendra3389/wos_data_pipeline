import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from autots import AutoTS
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score
import pickle
import warnings
import os
warnings.simplefilter('ignore')

df_22 = pd.read_excel(r"C:\Users\Programming.com\Desktop\08_02_24\SalesG_2022.xlsx")
df_23 = pd.read_excel(r"C:\Users\Programming.com\Desktop\08_02_24\Sales.xlsx")
df_24 = pd.read_excel(r"C:\Users\Programming.com\Downloads\Sales Order.xlsx",sheet_name='Sheet2',skiprows=0)
df_24_mar = pd.read_excel(r"C:\Users\Programming.com\Downloads\s.xlsx")
df_24_mar = df_24_mar[['OrderDate','m2 sold']].copy()
# Step 0: Load time series data into DataFrame
def load_time_series_data(df_22,df_23,df_24,df_24_mar):
    df_concatenated = pd.concat([df_22[['InvoiceDate', 'm² Sold']], 
                             df_23[['InvoiceDate', 'm² Sold']],
                             df_24.rename(columns={'OrderDate': 'InvoiceDate','m2 sold' : 'm² Sold'})[['InvoiceDate', 'm² Sold']],
                             df_24_mar.rename(columns={'OrderDate': 'InvoiceDate','m2 sold' : 'm² Sold'})[['InvoiceDate', 'm² Sold']]],
                            ignore_index=True)

    df_concatenated.dropna(subset=['InvoiceDate'],inplace=True)
    df_concatenated.InvoiceDate = pd.to_datetime(df_concatenated.InvoiceDate)
    df_concatenated['Week'] = df_concatenated.InvoiceDate.dt.to_period('B')
    df_concatenated.Week = df_concatenated.Week.dt.to_timestamp()
    df_22_24 = df_concatenated[['Week','m² Sold']].copy()
    df_22_24.Week = pd.to_datetime(df_22_24.Week)
    df_22_24 = df_22_24.set_index('Week')
    df_22_24_groupby = df_22_24.resample('W-Thu').sum()

    return df_22_24_groupby

# Step 1: Preprocess time series data
def preprocess_time_series_data(df):
    start_date = df.index.min()
    end_date = df.index.max()
    complete_date_range = pd.date_range(start=start_date, end=end_date, freq='W-Thu')
    df_reindexed = df.reindex(complete_date_range)
    missing_days = df_reindexed[df_reindexed.isnull().any(axis=1)].index

    if len(missing_days) == 0:
        df_Weekly = df.fillna(0)

        return df_Weekly
    else:
        print(f"after checking missing dates there is a list of valuese {missing_days}")

# Step 2: splitting_data_set
def splitting_data_set(df):

    train_size = int(len(df)*0.8)
    train,test = df.iloc[:train_size,:],df.iloc[train_size:,:]
    
    return train, test


        
###########################################################################################################
###########################################################################################################
p_values = [1, 2]  # Adjust as needed
d_values = [0,1]        # Adjust as needed
q_values = [1, 2]  # Adjust as needed


P_values = [0, 1]  # Adjust as needed
D_values = [0,1]        # Adjust as needed
Q_values = [0,1]  # Adjust as needed
s_values = [52]       # Seasonality, adjust based on your data
###########################################################################################################
hyperparameters = product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)


###########################################################################################################
###########################################################################################################


# Step 3: Train a machine learning model
# step3.1
def sarima_train_model(df,hyperparameters,train,test):

    # Initialize variables to store the best model and its performance
    best_rmse = float('inf')
    best_model = None
    best_order = None
    best_sessonal_order = None
    best_r2_score = None

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

    
    folder_path = r'C:\Users\Programming.com\Desktop\ff\VS\models'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    file_path = os.path.join(f"{folder_path}, 'sarima{best_model.model.order + best_model.model.seasonal_order}_{current_date}.pkl")
    with open(file_path,'wb')as file:
        pickle.dump(best_model, file)

    return best_model,best_rmse,best_r2_score,test

# step3.2
def run_arima(df,train,test):

    Auto_arima_model = auto_arima(df, start_p=0, d=0,start_q=0, max_p=5,max_d=3,max_q=5,
                   m=52, seasonal=True, trace=True,
                   start_P=0,D=0,start_Q=0,max_P=4,max_D=3,max_Q=4,
                   error_action='ignore', suppress_warnings=True, stepwise=True,
                   max_order=None,scoring='mse',
                   trend=None, with_intercept=True,
                   sarimax_kwargs=None, information_criterion='aic',
                   maxiter=50, disp=150, callback=None, offset_test_args=None,
                   seasonal_test_args=None, suppress_stdout=False)
    
    folder_path = r'C:\Users\Programming.com\Desktop\ff\VS\models'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    file_path = os.path.join(f"{folder_path}, 'Auto_arima_{current_date}.pkl")
    with open(file_path,'wb')as file:
        pickle.dump(Auto_arima_model, file)

    return Auto_arima_model

# step3.3
def run_Auto_ts(df,forecast_length,train,test):

    model = AutoTS(forecast_length=forecast_length,
                    frequency='W-Thu',
                    prediction_interval=0.95,
                    no_negatives=True,
                    ensemble=None,
                    models_mode='deep',
                    model_list='superfast',  # "superfast", "default", "fast_parallel","fast",'univariate
                    transformer_list="auto",  # "superfast",
                    #drop_most_recent=None,
                    max_generations=5,
                    num_validations=2,
                    validation_method='backwards',
                    n_jobs='auto')
    
    autots_model_fit = model.fit(df)

    folder_path = r'C:\Users\Programming.com\Desktop\ff\VS\models'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    file_path = os.path.join(f"{folder_path}, 'Auto_TS_{current_date}.pkl")
    with open(file_path,'wb')as file:
        pickle.dump(autots_model_fit, file)
                    
    return autots_model_fit


# step3.4
def run_phrophet(df):
    # 1. Preprocessing (ensure 'ds' is datetime format)
    df_p = df.reset_index()
    df_p = df_p.rename(columns={'Week': 'ds','m² Sold':'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'])  # Assuming 'ds' is a string column with dates

    train_size = int(len(df_p)*0.8)
    train_P,test_P = df_p.iloc[:train_size,:],df_p.iloc[train_size:,:]
    p_model = Prophet()

    prophet_model = p_model.fit(df_p)

    folder_path = r'C:\Users\Programming.com\Desktop\ff\VS\models'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    file_path = os.path.join(f"{folder_path},Prophet_{current_date}.pkl")
    with open(file_path,'wb')as file:
        pickle.dump(prophet_model, file)

    return prophet_model,test_P




# Step 4: Evaluate the model
def sarima_evaluate_model(best_model,best_rmse, best_r2_score,test,model_choice):
    if model_choice.lower() == 'sarima':
        print(f'Root_Mean Squared Error: {best_rmse}')
        print(f'Root_Mean Squared Error: {best_r2_score}')

        sarima_test = best_model.get_prediction(start=test.index.min(), end=test.index.max())

        sarima_test = sarima_test.predicted_mean
        mse = mean_squared_error(test['m² Sold'],sarima_test)
        print(f'Mean Squared Error on Test Set: {mse}')
        print(f"RMSE: {np.sqrt(mse)}")
        Sarima_r2score = r2_score(test['m² Sold'],sarima_test)
        print(f'r2_score on Test Set: {Sarima_r2score}')
        SMAPE = mean_absolute_percentage_error(test['m² Sold'],sarima_test)
        print(f'MAPE on Test Set: {SMAPE}')

        return sarima_test
    
def autoarima_evaluate_model(df,test,model_choice,Auto_arima_model):
    if model_choice.lower() == 'autoarima':
        
        report_arima = df.copy()
        report_arima.reset_index(inplace=True)

        arima_pred = Auto_arima_model.predict_in_sample(start=test.index.min(), end=test.index.max())
        arima_pred = pd.DataFrame(arima_pred).reset_index()
        arima_pred.columns = ['Week','arima_test_validation']
        report_arima = pd.merge(report_arima,arima_pred,on='Week',how='outer')

        print(arima_pred)
        print(test)
        RMSE = np.sqrt(mean_squared_error(test['m² Sold'], arima_pred['arima_test_validation']))
        MSE = mean_squared_error(test['m² Sold'], arima_pred['arima_test_validation'])
        r2 = r2_score(test['m² Sold'], arima_pred['arima_test_validation'])
        print(f'Mean Squared Error on Test Set: {MSE}')
        print(f'Root_Mean Squared Error on Test Set: {RMSE}')
        print(f'r2_score on Test Set: {r2}')

        return report_arima
    
def autots_evaluate_model(df,model_choice,autots_model_fit):      
    if model_choice.lower() == 'autots':
        validat_result = autots_model_fit.retrieve_validation_forecasts().reset_index()
        validat_result.columns = ['Week','PredictionInterval','SeriesID','ValidationRound','valid_values']
        validat_result = validat_result[(validat_result["PredictionInterval"] == '50%') & (validat_result['ValidationRound']=='0')]

        reporrt_auto_TS = df.copy()
        reporrt_auto_TS = reporrt_auto_TS.reset_index()
        reporrt_auto_TS = pd.merge(reporrt_auto_TS,validat_result,on='Week',how='inner')
        # reporrt_auto_TS.to_excel('validate.xlsx',index=False)
        reporrt_auto_TS_p98 = reporrt_auto_TS[(reporrt_auto_TS["PredictionInterval"] == '50%') & (reporrt_auto_TS['ValidationRound']=='0')]
        r2_auto_ts = r2_score(reporrt_auto_TS_p98['m² Sold'],reporrt_auto_TS_p98['valid_values'])
        mse = mean_squared_error(reporrt_auto_TS_p98['m² Sold'],reporrt_auto_TS_p98['valid_values'])
        print(f'Mean Squared Error on Test Set: {mse}')
        print(f"RMSE: {np.sqrt(mse)}")
        print(f'r2_score on Test Set: {r2_auto_ts}')
        SMAPE = mean_absolute_percentage_error(reporrt_auto_TS_p98['m² Sold'],reporrt_auto_TS_p98['valid_values'])
        print(f'MAPE on Test Set: {SMAPE}')

        return reporrt_auto_TS
    
def prophet_evaluate_model(df,test_P,model_choice,prophet_model,num_steps ):
    if model_choice.lower() == 'prophet':
            
        future = prophet_model.make_future_dataframe(periods=num_steps,freq='W-Thu')
        # print(future)

        p_forecast = prophet_model.predict(future)
        predictions_prophet = p_forecast[['ds','yhat']]

        # print(test_P)
        # print("++"*60)
        # print(predictions_prophet)
        prophet_r2_score = r2_score(test_P['y'],predictions_prophet[-len(test_P):][['yhat']])
        prophet_rmse = np.sqrt(mean_squared_error(test_P['y'],predictions_prophet[-len(test_P):][['yhat']]))
        prophet_mse = mean_squared_error(test_P['y'],predictions_prophet[-len(test_P):][['yhat']])
        predictions_prophet_df = predictions_prophet.rename(columns={'ds': 'Week', 'yhat': 'prediction_m² Sold'})
        print(f'Mean Squared Error on Test Set: {prophet_mse}')
        print(f"RMSE: {prophet_rmse}")
        print(f'r2_score on Test Set: {prophet_r2_score}')
        # SMAPE = mean_absolute_percentage_error(test_P['y'],predictions_prophet[-len(test_P):][['yhat']],)
        # print(f'MAPE on Test Set: {SMAPE}')
        # print(predictions_prophet_df)
        report_prophet = df.copy()
        report_prophet.reset_index(inplace=True)
        predictions_prophet_report = pd.merge(report_prophet,predictions_prophet_df,on='Week',how='inner')
        return predictions_prophet_report










def Sarima_forecast_future(df,best_model,evaluation,num_steps):

    print(best_model.summary())
    reports = df.copy()
    reports['sarima_test_predict'] = evaluation
    # print(reports)
    # reports = reports.set_index('Week')

    
    forecat = best_model.get_forecast(steps=num_steps)
    forecat = forecat.predicted_mean.values
    # Create datetime index for forecasted values
    last_date = reports.index[-1]
    start_date = last_date + pd.DateOffset(weeks=1)  # Add one month to last date
    forecast_dates = pd.date_range(start=start_date, periods=num_steps, freq='W-Thu')
    forecast_df = pd.DataFrame(forecat, index=forecast_dates, columns=['Sarima_Forecast'])
    sarima_future_forecast = pd.concat([reports,forecast_df])
    
    return sarima_future_forecast
#####################################################################################
def autots_forecast_future(autots_model_fit,evaluation,num_steps):


    prediction = autots_model_fit.predict(forecast_length=num_steps)
    forecasts = prediction.forecast.reset_index()
    forecasts.columns = ['Week','future_m² Sold']

    auto_TS_forcast_future = pd.merge(evaluation,forecasts,on='Week',how='outer')

    return auto_TS_forcast_future

###############################################################################################  
def autoarima_forecast_future(Auto_arima_model,evaluation):

    arima_pred = Auto_arima_model.predict(6)
    arima_pred = pd.DataFrame(arima_pred).reset_index()
    arima_pred.columns = ['Week','arima_future_prediction']
    arima_pred

    report_arima = pd.merge(evaluation,arima_pred,on='Week',how='outer')
    return report_arima
    
def prophet_forecast_future(evaluation):
    
    return evaluation
        



# Step 6: Execute the pipeline
def main_upto_evaluate(model_choice,num_steps):
    # file_path = 'time_series_data.csv'  # Replace with your file path
    df = load_time_series_data(df_22,df_23,df_24,df_24_mar)
    print('Data ingetion_completed',df)
    df_preprocessed = preprocess_time_series_data(df)
    print('Data preprocessing completed',df_preprocessed)
    train,test = splitting_data_set(df=df_preprocessed)
        
    if model_choice.lower() == 'sarima':
        best_model,best_rmse,best_r2_score,test = sarima_train_model(df_preprocessed,hyperparameters,train=train,test=test)
        print('sarima model training completed')
        # return best_model,best_rmse,best_r2_score,test
    elif model_choice.lower() == 'autoarima':
        Auto_arima_model = run_arima(df_preprocessed,train=train,test=test)
        # return Auto_arima_model
    elif model_choice.lower() == 'autots':
        autots_model_fit = run_Auto_ts(df_preprocessed,forecast_length=num_steps,train=train,test=test)
        # return autots_model_fit
    elif model_choice.lower() == 'prophet':
        prophet_model,test_P = run_phrophet(df=df_preprocessed)
        print('Phrophet_training completed')
        print(test_P)
        # return prophet_model
    else:
        print("Invalid model choice.")
        # return None


    if model_choice.lower()=='sarima':
        sarima_evaluate = sarima_evaluate_model(best_model,best_rmse,best_r2_score,test,model_choice)
        sarima_report = Sarima_forecast_future(best_model=best_model,evaluation=sarima_evaluate,
                                                                 df=df_preprocessed,num_steps=num_steps)
        sarima_report.to_excel("sarima_future_forecast.xlsx")
        print('report dumped')
        return sarima_evaluate,sarima_report
    elif model_choice.lower()=='autoarima':    
        autoarima_evaluate = autoarima_evaluate_model(df_preprocessed,test,model_choice,Auto_arima_model)
        Auto_arima_model = autoarima_forecast_future(Auto_arima_model=Auto_arima_model,evaluation=autoarima_evaluate)
        Auto_arima_model.to_excel("Auto_arima_model_future_forecast.xlsx")
        print('report dumped')
        return autoarima_evaluate,Auto_arima_model
    elif model_choice.lower()=='autots':  #num_steps,autots_model_fit,reporrt_auto_TS
        autots_evaluate = autots_evaluate_model(df_preprocessed,model_choice,autots_model_fit)
        autots_model_report = autots_forecast_future(autots_model_fit=autots_model_fit,evaluation=autots_evaluate,num_steps=num_steps)
        autots_model_report.to_excel("autots_model_report_future_forecast.xlsx")
        print('report dumped')
        return autots_evaluate,autots_model_report
    elif model_choice.lower()=='prophet':
        prophet_evaluate = prophet_evaluate_model(df_preprocessed,test_P,model_choice,prophet_model,num_steps)
        prophet_model_report = prophet_forecast_future(evaluation=prophet_evaluate)
        print(prophet_model_report)
        prophet_model_report.to_excel("prophet_model_report_future_forecast.xlsx",index=False)
        print('report dumped')
        return prophet_evaluate,prophet_model_report
    else:
        print('Invalid model name')


if __name__ == "__main__":
    model_choice = input("Select a model (AutoARIMA, AutoTS, SARIMA, Prophet): ")
    main_upto_evaluate(model_choice,num_steps=5)


