# app.py
# -*- coding: utf-8 -*-
import io
import os
import base64
from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---- import your forecasting utilities exactly as in your Streamlit app ----
from process import (
    recursive_forecast, arima_forecast, sarimax_forecast, prophet_forecast,
    prophet_data_formatter, plot_model_forecast, model_params,
    download_entity_data, training_data_processor, model_mapes, prophet_mapes
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "scaler")

# New (local data folder)
PROPHET_SALES_PATH = os.path.join("data", "prophet_forecasts_sales.csv")
PROPHET_ORDERS_PATH = os.path.join("data", "prophet_forecasts_orders.csv")

try:
    PROPHET_SALES_DF = pd.read_csv(PROPHET_SALES_PATH)
    PROPHET_ORDERS_DF = pd.read_csv(PROPHET_ORDERS_PATH)
except Exception as e:
    # Fallback empty frames if local files not found
    print(f"[Warning] Failed to load local Prophet forecasts: {e}")
    PROPHET_SALES_DF = pd.DataFrame()
    PROPHET_ORDERS_DF = pd.DataFrame()

# ---------- Helpers ----------
ENTITIES = ["Company", "Region 1", "Region 2", "Region 3", "Region 4"]
TARGETS = ["Sales", "Orders"]
MODELS  = ["Linear Regression", "XGBoost", "ARIMA", "SARIMAX", "Prophet"]

def fig_to_base64_png(fig) -> str:
    """Convert a Matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def df_to_html_table(df: pd.DataFrame) -> str:
    return df.reset_index().to_html(
        classes="table table-striped table-hover align-middle small",
        border=0,
        index=False
    )

def build_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "entities": ENTITIES,
        "targets": TARGETS,
        "models": MODELS,
        "default_horizon": 30
    }

    if request.method == "GET":
        return render_template("index.html", **context)

    # ------- Handle POST (Run Forecast) -------
    entity = request.form.get("entity")
    target_choice = request.form.get("target")
    model_choice = request.form.get("model")
    try:
        m_steps = int(request.form.get("horizon", 30))
    except ValueError:
        m_steps = 30

    # Validate selections
    if entity not in ENTITIES or target_choice not in TARGETS or model_choice not in MODELS:
        context["error"] = "Invalid selection. Please try again."
        return render_template("index.html", **context)

    # Step 2: Load Entity-Specific Data
    ts_data, exog_data = download_entity_data(entity)

    # Step 3: Run Forecast
    forecast = None
    error = None
    try:
        if model_choice in ["Linear Regression", "XGBoost"]:
            ts_proc = training_data_processor(ts_data.copy(), target_col=target_choice)
            exog_pred = exog_data.head(m_steps)
            model_flag = 'lr' if model_choice == "Linear Regression" else 'xgb'
            forecast = recursive_forecast(ts_proc, exog_pred, model=model_flag, target_col=target_choice)

        elif model_choice == "ARIMA":
            forecast = arima_forecast(
                ts_data.copy(),
                m_steps,
                model_params[entity]['arima_order'],
                target_col=target_choice
            )

        elif model_choice == "SARIMAX":
            ts_proc = training_data_processor(ts_data.copy(), target_col=target_choice)
            exog_train = ts_proc[["Holiday", "Discounted Stores"]]
            exog_pred = exog_data[["Holiday", "Discounted Stores"]].head(m_steps)
            forecast = sarimax_forecast(
                ts_proc,
                m_steps,
                exog_train,
                exog_pred,
                model_params[entity]['sarimax_order'],
                model_params[entity]['seasonal_order'],
                target_col=target_choice
            )

        elif model_choice == "Prophet":
            if target_choice == "Sales":
                df = PROPHET_SALES_DF.copy()
                col_name = f"{entity}_Sales"
            else:
                df = PROPHET_ORDERS_DF.copy()
                col_name = f"{entity}_Orders"

            if df.empty or col_name not in df.columns:
                raise RuntimeError("Precomputed Prophet data unavailable for this selection.")

            forecast = df.loc[:m_steps-1, ["Date", col_name]].copy()
            forecast = forecast.rename(columns={col_name: target_choice})
            forecast["Date"] = pd.to_datetime(forecast["Date"])
            forecast = forecast.set_index("Date")

    except Exception as e:
        error = f"Forecasting failed: {e}"

    if error:
        context["error"] = error
        return render_template("index.html", **context)

    # Step 4: Plot Forecast
    fig = plot_model_forecast(
        ts_data.copy(),
        forecast,
        model_name=model_choice,
        inf_label=entity,
        target_col=target_choice
    )
    plot_b64 = fig_to_base64_png(fig)

    # Step 4.1: MAPE message
    mape_info = {"header": None, "lines": []}
    if model_choice == "Prophet":
        entity_key = entity
        if target_choice == "Sales":
            mape = (prophet_mapes.get(entity_key, {}) or {}).get('sales_mape')
            if mape is not None:
                mape_info["header"] = f"Test MAPE: {mape:.2%}"
                mape_info["lines"] = [
                    "Sales forecasts for Prophet are pre-computed.",
                    f"Best MAPE: {mape:.2%} on a 61-day horizon.",
                    "Forecasting with other horizons noticeably reduces performance.",
                    "Lower MAPE indicates better accuracy."
                ]
        else:
            mape = (prophet_mapes.get(entity_key, {}) or {}).get('orders_mape')
            if mape is not None:
                mape_info["header"] = f"Test MAPE: {mape:.2%}"
                mape_info["lines"] = [
                    "Order forecasts for Prophet are pre-computed.",
                    f"MAPE: {mape:.2%} on a 61-day horizon.",
                    "Forecasting with other horizons noticeably reduces performance.",
                    "Lower MAPE indicates better accuracy."
                ]
    else:
        key = (entity, target_choice, model_choice)
        if target_choice == "Sales":
            mape_value = model_mapes.get(key)
            if mape_value:
                mape_info["header"] = f"Test MAPE: {mape_value}"
                mape_info["lines"] = [
                    f"{model_choice} achieved this MAPE on a 61-day test horizon.",
                    "Lower MAPE indicates better accuracy.",
                    "Parameters tuned for minimum MAPE were used."
                ]
        else:
            mape_sales_key = (entity, "Sales", model_choice)
            mape_value = model_mapes.get(mape_sales_key)
            if mape_value:
                mape_info["header"] = f"Orders forecasts for {model_choice} are untested."
                mape_info["lines"] = [
                    f"Sales MAPE for the same model/segment: {mape_value}.",
                    "Lower MAPE indicates better accuracy."
                ]

    # Step 5: Table + CSV (provide a client-side download via data URL or blob)
    table_html = df_to_html_table(forecast)
    csv_bytes = build_csv_bytes(forecast)
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")

    # Render results
    return render_template(
        "index.html",
        **context,
        submitted=True,
        entity=entity,
        target_choice=target_choice,
        model_choice=model_choice,
        horizon=m_steps,
        plot_b64=plot_b64,
        mape_info=mape_info,
        table_html=table_html,
        csv_b64=csv_b64
    )


if __name__ == "__main__":
    # Run:  python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
