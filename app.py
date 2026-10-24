import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="FinPulse",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    
    with open("model_info.json", "r") as f:
        model_info = json.load(f)

    return model, scaler, feature_cols, model_info


model, scaler, feature_cols, model_info = load_model()


# =====================================================
# DOWNLOAD LATEST MARKET DATA
# =====================================================

@st.cache_data(ttl=3600)
def get_market_data():

    spy = yf.download(
        "SPY",
        period="6mo",
        auto_adjust=True,
        progress=False
    )

    vix = yf.download(
        "^VIX",
        period="6mo",
        auto_adjust=True,
        progress=False
    )

    # Flatten yfinance column structure
    spy.columns = spy.columns.get_level_values(0)
    vix.columns = vix.columns.get_level_values(0)

    # Keep only VIX Close
    vix_close = vix[["Close"]].copy()
    vix_close.columns = ["VIX"]

    # Join SPY and VIX using trading date
    data = spy.join(
        vix_close,
        how="left"
    )

    # Drop the last row if it is a partial candle for the current trading day
    if len(data) > 0:
        latest_bar_date = pd.to_datetime(data.index[-1]).date()
        today = pd.Timestamp.now().date()
        if latest_bar_date >= today:
            data = data.iloc[:-1].copy()

    return data


data = get_market_data()


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Daily log return
data["Return"] = (
    100 *
    np.log(
        data["Close"] /
        data["Close"].shift(1)
    )
)

# Absolute return = volatility proxy
data["Abs_Return"] = data["Return"].abs()


# -----------------------------
# Rolling volatility
# -----------------------------

data["Vol_5"] = (
    data["Return"]
    .rolling(5)
    .std()
)

data["Vol_10"] = (
    data["Return"]
    .rolling(10)
    .std()
)

data["Vol_21"] = (
    data["Return"]
    .rolling(21)
    .std()
)


# -----------------------------
# Return lags
# -----------------------------

data["Return_Lag1"] = data["Return"].shift(1)
data["Return_Lag2"] = data["Return"].shift(2)
data["Return_Lag3"] = data["Return"].shift(3)
data["Return_Lag5"] = data["Return"].shift(5)


# -----------------------------
# Absolute-return lags
# -----------------------------

data["AbsReturn_Lag1"] = data["Abs_Return"].shift(1)
data["AbsReturn_Lag2"] = data["Abs_Return"].shift(2)
data["AbsReturn_Lag5"] = data["Abs_Return"].shift(5)


# -----------------------------
# VIX features
# -----------------------------

data["VIX_Lag1"] = data["VIX"].shift(1)

data["VIX_Change_1D"] = (
    data["VIX"].diff(1)
)

data["VIX_Change_5D"] = (
    data["VIX"].diff(5)
)


# -----------------------------
# Parkinson volatility
# -----------------------------

data["Parkinson"] = np.sqrt(
    (
        np.log(
            data["High"] /
            data["Low"]
        ) ** 2
    )
    /
    (4 * np.log(2))
)


# -----------------------------
# Volume change
# -----------------------------

data["Volume_Change_1D"] = (
    data["Volume"]
    .pct_change() * 100
)


# =====================================================
# GENERATE FORECAST
# =====================================================

features = data[feature_cols].dropna()

latest_features = features.iloc[[-1]]

# t = latest trading day with all required features
latest_date = latest_features.index[-1]

# Apply scaler only if the selected model requires scaled inputs (Ridge and SVR do, XGBoost does not)
needs_scaling = model_info.get("needs_scaling", True)

if needs_scaling:
    latest_input = scaler.transform(latest_features)
else:
    latest_input = latest_features

prediction = model.predict(
    latest_input
)[0]

# Volatility cannot be negative
prediction = max(float(prediction), 0.0)


# =====================================================
# CURRENT MARKET VALUES
# =====================================================

latest_row = data.loc[latest_date]

latest_price = float(
    latest_row["Close"]
)

latest_vix = float(
    latest_row["VIX"]
)

latest_return = float(
    latest_row["Return"]
)

latest_abs_return = float(
    latest_row["Abs_Return"]
)

latest_vol_21 = float(
    latest_row["Vol_21"]
)


# Approximate movement in dollar terms
expected_move_dollars = (
    latest_price *
    prediction / 100
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📈 FinPulse")

    st.write(
        "Short-term financial market "
        "volatility forecasting using SPY and VIX."
    )

    if st.button(
        "🔄 Refresh Market Data",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.markdown("**Asset**")
    st.write("SPY")

    st.markdown("**Model**")
    st.write(model_info.get("model_name", "Ridge Regression"))

    st.markdown("**Forecast Horizon**")
    st.write("Next trading session")

    st.markdown("**Data Source**")
    st.write("Yahoo Finance")

    st.divider()

    st.caption(
        "Educational forecasting project. "
        "Not financial advice."
    )


# =====================================================
# HEADER
# =====================================================

st.title("📈 FinPulse")

st.markdown(
    "### Financial Market Volatility Forecasting Engine"
)

st.caption(
    "Forecasting the magnitude of SPY's next-session "
    "market movement using recent SPY and VIX information."
)


# =====================================================
# FORECAST CONTEXT
# =====================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Asset",
    "SPY"
)

c2.metric(
    "Latest Market Data",
    latest_date.strftime("%d %b %Y")
)

c3.metric(
    "Forecast For",
    "Next Trading Session"
)


st.caption(
    "The latest market date is treated as time t. "
    "FinPulse forecasts the next trading session, t+1."
)


st.divider()


# =====================================================
# MAIN FORECAST
# =====================================================

st.subheader("🔮 Next-Session Volatility Forecast")

# Model output represents expected absolute return E|R|
norm_scaling = np.sqrt(2.0 / np.pi)
sigma_est = prediction / norm_scaling  # sigma = E|R| / sqrt(2/pi)
var95_pct = 1.6449 * sigma_est
var95_dollars = latest_price * (var95_pct / 100.0)

f1, f2, f3 = st.columns(3)

f1.metric(
    "Expected Absolute Return E[|R|]",
    f"{prediction:.3f}%"
)

f2.metric(
    "Estimated Std. Dev. (σ = E[|R|]/√(2/π))",
    f"{sigma_est:.3f}%"
)

f3.metric(
    "95% Parametric VaR (1-Day)",
    f"±{var95_pct:.2f}% (${var95_dollars:.2f})"
)

st.info(
    f"**Model Forecast Interpretation:** FinPulse's production model ({model_info.get('model_name', 'Ridge Regression')}) "
    f"directly predicts next-session expected absolute return: **E[|R_{{t+1}}|] = {prediction:.3f}%** "
    f"($${expected_move_dollars:.2f} movement per share at current price $${latest_price:.2f}). "
    f"Under the standard Gaussian distribution r ~ N(0, σ²), this translates to an estimated conditional standard deviation of "
    f"**σ = {sigma_est:.3f}%** (via σ = E[|R|] / √(2/π)). "
    f"Under this calibrated variance, the 1-day 95% Value-at-Risk threshold is **{var95_pct:.2f}%** ($${var95_dollars:.2f}/share)."
)


# =====================================================
# CURRENT MARKET SNAPSHOT
# =====================================================

st.subheader("Current Market Snapshot")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "SPY Price",
    f"${latest_price:.2f}"
)

m2.metric(
    "Latest SPY Return",
    f"{latest_return:.3f}%"
)

m3.metric(
    "21-Day Volatility",
    f"{latest_vol_21:.3f}%"
)

m4.metric(
    "VIX",
    f"{latest_vix:.2f}"
)


st.caption(
    "VIX reflects expected near-term market volatility. "
    "Higher values generally indicate greater expected "
    "market uncertainty and larger potential movements."
)


st.divider()


# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "📊 Market Overview",
        "📈 Volatility"
    ]
)


# =====================================================
# TAB 1 — MARKET OVERVIEW
# =====================================================

with tab1:

    st.subheader("SPY Closing Price")

    st.caption(
        "Recent SPY price movement over the market period "
        "used to generate the latest forecast."
    )

    fig_price = go.Figure()

    fig_price.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="SPY Price"
        )
    )

    # Highlight latest observation
    fig_price.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_price],
            mode="markers",
            marker=dict(size=10),
            name="Latest Observation (t)"
        )
    )

    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True
    )


    # ---------------------------------
    # MARKET SUMMARY
    # ---------------------------------

    st.subheader("Latest Market Observation")

    summary = pd.DataFrame({
        "Measure": [
            "SPY Closing Price",
            "SPY Daily Return",
            "Absolute Return",
            "21-Day Volatility",
            "VIX"
        ],

        "Value": [
            f"${latest_price:.2f}",
            f"{latest_return:.3f}%",
            f"{latest_abs_return:.3f}%",
            f"{latest_vol_21:.3f}%",
            f"{latest_vix:.2f}"
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )


# =====================================================
# TAB 2 — VOLATILITY
# =====================================================

with tab2:

    # ---------------------------------
    # ABSOLUTE RETURNS
    # ---------------------------------

    st.subheader(
        "Observed Daily Volatility Proxy"
    )

    st.caption(
        "FinPulse uses the absolute value of SPY's daily "
        "return as a simple volatility proxy. "
        "Higher spikes indicate larger daily market movements."
    )

    fig_abs = go.Figure()

    fig_abs.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Abs_Return"],
            mode="lines",
            name="Absolute Return"
        )
    )

    fig_abs.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_abs_return],
            mode="markers",
            marker=dict(size=10),
            name="Latest Observation (t)"
        )
    )

    fig_abs.update_layout(
        xaxis_title="Date",
        yaxis_title="Absolute Return (%)",
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig_abs,
        use_container_width=True
    )


    # ---------------------------------
    # ROLLING VOLATILITY
    # ---------------------------------

    st.subheader(
        "Recent Historical Volatility"
    )

    st.caption(
        "Rolling volatility shows how variable SPY returns "
        "have been over different recent time windows."
    )

    fig_rolling = go.Figure()

    fig_rolling.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Vol_5"],
            mode="lines",
            name="5-Day Volatility"
        )
    )

    fig_rolling.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Vol_10"],
            mode="lines",
            name="10-Day Volatility"
        )
    )

    fig_rolling.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Vol_21"],
            mode="lines",
            name="21-Day Volatility"
        )
    )

    fig_rolling.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig_rolling,
        use_container_width=True
    )


    # ---------------------------------
    # VIX
    # ---------------------------------

    st.subheader(
        "Market-Implied Volatility — VIX"
    )

    st.caption(
        "VIX reflects the market's expectation of "
        "near-term S&P 500 volatility. "
        "Higher values generally indicate greater "
        "expected uncertainty."
    )

    fig_vix = go.Figure()

    fig_vix.add_trace(
        go.Scatter(
            x=data.index,
            y=data["VIX"],
            mode="lines",
            name="VIX"
        )
    )

    fig_vix.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_vix],
            mode="markers",
            marker=dict(size=10),
            name="Latest Observation (t)"
        )
    )

    fig_vix.update_layout(
        xaxis_title="Date",
        yaxis_title="VIX",
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig_vix,
        use_container_width=True
    )


    # ---------------------------------
    # SIMPLE EXPLANATION
    # ---------------------------------

    with st.expander(
        "What does volatility mean?"
    ):

        st.write(
            "Volatility describes how strongly market prices "
            "or returns move over time."
        )

        st.write(
            "Small and stable movements indicate relatively "
            "low volatility, while large positive or negative "
            "movements indicate higher volatility."
        )

        st.write(
            "FinPulse predicts the magnitude of the next "
            "session's SPY return rather than its direction."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "FinPulse • Market data retrieved through Yahoo Finance "
    "using yfinance • Educational forecasting project • "
    "Not financial advice."
)