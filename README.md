🛒 Product Sales Forecasting
📖 Project Overview

This project builds and deploys a sales forecasting system for retail stores using historical transaction data.
The dataset covers 365 stores across 4 regions and 5 location types from Jan 2018 – May 2019.

The goal is to:

Predict daily sales and orders at company and region levels.

Identify the impact of discounts, holidays, and store characteristics on sales.

Provide a web app for generating forecasts interactively.

🔍 Key Insights from EDA

Store Type S4 had the highest average sales per record, though less frequent.

Region R1 was the strongest performer in both sales and orders.

Discounts boost sales significantly (~+12K increase per record).

Holidays reduce sales (~−9K compared to regular days).

Sales showed strong weekly and seasonal patterns.

⚙️ Modeling Approach

The project applied both classical time-series and machine learning models:

Linear Regression (LR) → Simple, interpretable, best short-term accuracy (MAPE ~6–7%).

XGBoost → Captured non-linear patterns, best long-term recursive forecasting (MAPE ~10–12%).

ARIMA → Good baseline (MAPE ~6.9%), weaker with holiday effects.

SARIMAX → Handled exogenous variables (holidays, discounts), improved to ~6.5% MAPE.

Prophet → Easy to interpret seasonal/holiday trends, accuracy comparable to ARIMA.

📊 Performance Snapshot:

Company-level: LR MAPE ~6.7%, XGB ~7.0%

Recursive (60 days): XGB ~10.8%, better than LR (~13–15%)

🖥️ Deployment Pipeline

Flask app built with a simple UI:

Select Entity (Company / Region)

Select Target (Sales / Orders)

Choose Model (LR, XGB, ARIMA, SARIMAX, Prophet)

Enter Forecast Horizon (e.g., 30 days)

Backend (process.py) loads data, applies preprocessing, generates forecasts.

Output: forecast plot, MAPE metric, and downloadable CSV.

Hosted on Heroku/Render for demo access.
