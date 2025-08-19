import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    mean_absolute_percentage_error as mape)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from itertools import product
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:,.2f}'.format)
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').disabled = True
import os
import urllib.request

# Downloading data
"""
    Download training and inference data for the given entity (Company or Region).
    Returns:
        ts_data: Time series training data (with 'Sales' and exogenous features)
        exog_data: Exogenous inference data 
    """

def download_entity_data(entity: str):
    
    base_url_train = "https://raw.githubusercontent.com/varunjoshua/ScalerDSML-ProductSalesForecast/refs/heads/main/data/"
    base_url_inf = "https://raw.githubusercontent.com/varunjoshua/ScalerDSML-ProductSalesForecast/refs/heads/main/data/"

    entity_map = {
        "Company": ("ts_co.csv", "inf_all.csv"),
        "Region 1": ("ts_r1.csv", "inf_r1.csv"),
        "Region 2": ("ts_r2.csv", "inf_r2.csv"),
        "Region 3": ("ts_r3.csv", "inf_r3.csv"),
        "Region 4": ("ts_r4.csv", "inf_r4.csv"),
    }

    train_file, inf_file = entity_map[entity]
    
    ts_data = pd.read_csv(base_url_train + train_file, parse_dates=[0], index_col=0)
    exog_data = pd.read_csv(base_url_inf + inf_file, parse_dates=[0], index_col=0)

    return ts_data, exog_data


"""# **Pre-processing functions**


## **Function to process data for Regression**
"""

#Processes training data for model training.

    #Parameters:
    #- ts_data (pd.DataFrame): DataFrame with datetime index and columns: 'Holiday', 'Discount', 'Discounted Stores', 'Orders', 'Sales'
    #- target_col (str): Target column to forecast, either 'Sales' or 'Orders'

    #Returns:
    #- ts_processed (pd.DataFrame): Processed DataFrame with:
        #- 'target' column
        #- required features for modeling

def training_data_processor(ts_data: pd.DataFrame, target_col: str = "Sales"):
    
    assert target_col in ["Sales", "Orders"], "target_col must be 'Sales' or 'Orders'"

    df = ts_data.copy()

    # Create time-based features
    df['Day Count'] = (df.index - df.index.min()).days
    df['Weekend'] = df.index.weekday.isin([5, 6]).astype(int)
    df['Month_sine'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cosine'] = np.cos(2 * np.pi * df.index.month / 12)
    df['Day of Week_sine'] = np.sin(2 * np.pi * df.index.weekday / 7)
    df['Day of Week_cosine'] = np.cos(2 * np.pi * df.index.weekday / 7)

    df = df.rename(columns={target_col: "target"})

    # Select relevant columns
    features = [
        "target", "Holiday", "Discounted Stores", "Day Count", "Weekend",
        "Month_sine", "Month_cosine", "Day of Week_sine", "Day of Week_cosine"
    ]
    df = df[features]

    return df



# Function to tranform and process the inference data given
  # The test data provided is ungrouped with records of all stores for each day
  # The data needs to be grouped and transformed for the Recursive Forecasting function
  # The function will group and aggregate the data for Company and Regions : R1, R2, R3, R4

def inference_data_processor(data):

    # Step 1: Convert 'Date' to datetime and add 'Discounted_Flag'
    data['Date'] = pd.to_datetime(data['Date'])
    data['Discounted_Flag'] = data['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Step 2: function to process each group
    def process_group(group_df):
        group_df = group_df.groupby('Date').agg({
            'Holiday': 'last',
            'Discounted_Flag': lambda x: x.sum() / x.count()
        }).rename(columns={'Discounted_Flag': 'Discounted Stores'})

        # Date features
        group_df['Day Count'] = (group_df.index - group_df.index.min()).days
        group_df['Weekend'] = group_df.index.dayofweek.isin([5, 6]).astype(int)
        day_of_week = group_df.index.dayofweek
        month = group_df.index.month

        # Cyclical features
        group_df['Month_sine'] = np.sin(2 * np.pi * month / 12)
        group_df['Month_cosine'] = np.cos(2 * np.pi * month / 12)
        group_df['Day of Week_sine'] = np.sin(2 * np.pi * day_of_week / 7)
        group_df['Day of Week_cosine'] = np.cos(2 * np.pi * day_of_week / 7)

        return group_df

    # Step 3: Create datasets
    inf_all = process_group(data)
    inf_r1 = process_group(data[data['Region_Code'] == 'R1'])
    inf_r2 = process_group(data[data['Region_Code'] == 'R2'])
    inf_r3 = process_group(data[data['Region_Code'] == 'R3'])
    inf_r4 = process_group(data[data['Region_Code'] == 'R4'])

    return inf_all, inf_r1, inf_r2, inf_r3, inf_r4

"""## **Function to process data for SARIMAX**"""

# Function inference_exog_processor will be used to transform and process the inference dataset...
#...and return datframes with exog variable for Company and Regions : R1, R2, R3, R4, for the inference period

def inference_exog_processor(data):

    # Step 1: Convert 'Date' to datetime and add 'Discounted_Flag'
    data['Date'] = pd.to_datetime(data['Date'])
    data['Discounted_Flag'] = data['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Step 2: function to process each group
    def process_group(group_df):
        group_df = group_df.groupby('Date').agg({
            'Holiday': 'last',
            'Discounted_Flag': lambda x: x.sum() / x.count()
        }).rename(columns={'Discounted_Flag': 'Discounted Stores'})

        return group_df

    # Step 3: Creating datasets
    exog_all = process_group(data)
    exog_r1 = process_group(data[data['Region_Code'] == 'R1'])
    exog_r2 = process_group(data[data['Region_Code'] == 'R2'])
    exog_r3 = process_group(data[data['Region_Code'] == 'R3'])
    exog_r4 = process_group(data[data['Region_Code'] == 'R4'])

    return exog_all, exog_r1, exog_r2, exog_r3, exog_r4

"""## **Function to process data for Prophet**"""

def prophet_data_formatter(data, is_inference=False):
    df = pd.DataFrame()
    data = data.copy()
    data = data.reset_index()
    data.rename(columns={data.columns[0]: 'Date'}, inplace=True)

    if not is_inference:
        df['y'] = data['Sales']

    df['ds'] = pd.to_datetime(data['Date'])
    exog_cols = ['Holiday', 'Discounted Stores']
    exog = data[exog_cols]
    df = pd.concat([df, exog.reset_index(drop=True)], axis=1)
    return df

"""# **Model Params**"""

model_params = {
    'Company': {
        'arima_order': (3, 0, 2),
        'sarimax_order': (2, 1, 2),
        'seasonal_order': (1, 0, 2, 7)
    },
    'Region 1': {
        'arima_order': (3, 1, 3),
        'sarimax_order': (2, 1, 1),
        'seasonal_order': (2, 1, 0, 7)
    },
    'Region 2': {
        'arima_order': (3, 1, 3),
        'sarimax_order': (0, 1, 2),
        'seasonal_order': (2, 1, 0, 7)
    },
    'Region 3': {
        'arima_order': (3, 0, 2),
        'sarimax_order': (0, 1, 1),
        'seasonal_order': (2, 1, 0, 7)
    },
    'Region 4': {
        'arima_order': (1, 1, 1),
        'sarimax_order': (2, 1, 2),
        'seasonal_order': (1, 0, 2, 7)
    }
}

print(model_params)

"""# **Model MAPEs**"""
# Model MAPEs for each entity and model type

model_mapes = {
    # Linear Regression (no exogenous)
    ("Company", "Sales", "Linear Regression"): "10.8%",
    ("Region 1", "Sales", "Linear Regression"): "14.4%",
    ("Region 2", "Sales", "Linear Regression"): "12.4%",
    ("Region 3", "Sales", "Linear Regression"): "15.1%",
    ("Region 4", "Sales", "Linear Regression"): "13.8%",

    # XGBoost (no exogenous)
    ("Company", "Sales", "XGBoost"): "12.8%",
    ("Region 1", "Sales", "XGBoost"): "12.8%",
    ("Region 2", "Sales", "XGBoost"): "10.1%",
    ("Region 3", "Sales", "XGBoost"): "11.5%",
    ("Region 4", "Sales", "XGBoost"): "11.7%",

    # ARIMA
    ("Company", "Sales", "ARIMA"): "14.36%",
    ("Region 1", "Sales", "ARIMA"): "15.90%",
    ("Region 2", "Sales", "ARIMA"): "14.26%",
    ("Region 3", "Sales", "ARIMA"): "15.78%",
    ("Region 4", "Sales", "ARIMA"): "15.85%",

    # SARIMAX
    ("Company", "Sales", "SARIMAX"): "11.47%",
    ("Region 1", "Sales", "SARIMAX"): "11.97%",
    ("Region 2", "Sales", "SARIMAX"): "9.88%",
    ("Region 3", "Sales", "SARIMAX"): "11.90%",
    ("Region 4", "Sales", "SARIMAX"): "12.78%",

    # Prophet
    ("Company", "Sales", "Prophet"): "10.12%",
    ("Region 1", "Sales", "Prophet"): "11.16%",
    ("Region 2", "Sales", "Prophet"): "8.83%",
    ("Region 3", "Sales", "Prophet"): "11.87%",
    ("Region 4", "Sales", "Prophet"): "12.78%",
}

prophet_mapes = {'Company': {'sales_mape': 0.1021, 'orders_mape': 0.1627},
 'Region 1': {'sales_mape': 0.1083, 'orders_mape': 0.175},
 'Region 2': {'sales_mape': 0.0872, 'orders_mape': 0.1361},
 'Region 3': {'sales_mape': 0.1185, 'orders_mape': 0.1578},
 'Region 4': {'sales_mape': 0.1302, 'orders_mape': 0.1876}}


"""# **Recursive Linear Regression Forecasting**"""

#Function performs recursive forecasting using Linear Regression or XGBoost.

#Parameters:
  # df_train: Processed training data (pandas dataframe) with target_col
  # df_inference: Processed inference data (no target_col required)
  # model_type: 'lr' for Linear Regression or 'xgb' for XGBoost
  # target_col: Target variable name (default='Sales')

#Returns:
  # Pandad DataFrame with forecasted Sales for the inference period


def recursive_forecast(df_train, df_inference, model='lr', target_col='Sales'):
    df_train = df_train.copy()
    df_inference = df_inference.copy()

    df_train.index.freq = 'D'
    df_inference.index.freq = 'D'

    # Always use 'target' as the column after processing
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target'])
    X_inference = df_inference.drop(columns=['target']) if 'target' in df_inference.columns else df_inference

    max_lag = 31
    window_features = RollingFeatures(
        stats=['mean', 'mean', 'mean'],
        window_sizes=[7, 14, 31]
    )

    if model == 'lr':
        forecaster = ForecasterRecursive(
            regressor=LinearRegression(),
            lags=[1, 2, 3, 7, 31],
            window_features=window_features
        )
    elif model == 'xgb':
        forecaster = ForecasterRecursive(
            regressor=XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            lags=[1, 2, 3, 7, 31]
        )
    else:
        raise ValueError("Invalid model type. Choose 'lr' or 'xgb'.")

    forecaster.fit(y=y_train, exog=X_train)
    y_pred = forecaster.predict(steps=len(df_inference), exog=X_inference, last_window=y_train[-max_lag:])

    df_forecast = df_inference.copy()
    df_forecast[target_col] = y_pred.values  # Use target_col instead of 'Sales'

    return df_forecast

"""# **ARIMA Forecasting**"""

# The arima_forecast function will
  # Train the model on given data using pre-computed best p,d,q order
  # Use model to forecast m steps in the future


def arima_forecast(df_train, m_steps, arima_order=(1, 1, 1), target_col='Sales'):
    df_train = df_train.copy()
    df_train.index.freq = 'D'

    # Fit ARIMA model
    model = ARIMA(df_train[target_col], order=arima_order)
    model_fit = model.fit()

    # Forecast for the inference period
    forecast = model_fit.forecast(steps=m_steps)

    # Prepare output DataFrame
    future_index = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=m_steps, freq='D')
    df_forecast = pd.DataFrame({target_col: forecast}, index=future_index)

    return df_forecast

"""# **SARIMAX Forecasting**"""

# The sarima_forecast function will
  # Train the model on given data using pre-computed best p,d,q,P,D,Q,s order
  # Use model to forecast m steps in the future

def sarimax_forecast(df_train, m_steps, exog_train, exog_pred,
                     order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), target_col='Sales'):

    df_train = df_train.copy()
    exog_train = exog_train.copy()
    exog_pred = exog_pred.copy()

    df_train.index.freq = 'D'
    exog_train.index.freq = 'D'
    exog_pred.index.freq = 'D'

    # Fit SARIMAX on full training data
    model = SARIMAX(df_train['target'],
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train)

    model_fit = model.fit(disp=False)

    # Forecast for the inference period
    forecast = model_fit.forecast(steps=m_steps, exog=exog_pred)

    # Prepare output DataFrame
    future_index = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=m_steps, freq='D')
    df_forecast = pd.DataFrame({target_col: forecast}, index=future_index)  # Use target_col

    return df_forecast

"""# **Prophet Forecasting**"""

def prophet_forecast(ts_data, m_steps, exog_pred, target_col='Sales'):
    import warnings
    warnings.filterwarnings("ignore")

    # ts_data contains: ['ds', 'y', 'Holiday', 'Discounted Stores']
    # exog_pred contains: ['ds', 'Holiday', 'Discounted Stores'] for m_steps days

    # Step 1: Fit Prophet model on full historical data
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=1.25,
        seasonality_mode='additive'
    )
    model.add_regressor('Holiday')
    model.add_regressor('Discounted Stores')
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(ts_data)

    # Step 2: Forecast m future steps using exog_pred
    forecast = model.predict(exog_pred)

    # Return DataFrame with 'Date' as index and target_col as column name
    df_forecast = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': target_col})
    df_forecast = df_forecast.set_index('Date')

    return df_forecast

"""# **Plot Function**"""

"""
    Plot historical and forecasted sales for a single model.

    Parameters:
    - df_train: DataFrame with 'Sales' and datetime index (training data)
    - df_forecast: DataFrame with 'Sales' and datetime index (forecasted values)
    - model_name: Name of the forecasting model (str)
    - inf_label: Optional label for the plot (e.g., 'Company' or 'Region')
    - target_col: Target column to plot (default='Sales')
    """

def plot_model_forecast(df_train, df_forecast, model_name, inf_label='', target_col='Sales'):
    # Plot historical and forecasted values for the specified target_col ("Sales" or "Orders")
    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical data (last 100 days)
    ax.plot(df_train.index[-100:], df_train[target_col][-100:], label=f'Historical {target_col}', color='black')

    # Forecast
    ax.plot(df_forecast.index, df_forecast[target_col], linestyle='--', label=f'Forecast ({model_name})', color='blue')

    ax.set_title(f"Forecasted {target_col} for {inf_label} ({model_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig