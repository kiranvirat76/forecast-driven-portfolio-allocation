# Forecast-Driven Portfolio Allocation

Quantitative time-series forecasting, volatility analysis, and portfolio optimization on NSE equities using ARIMA, Prophet, ensemble modelling, and walk-forward validation.

---

# Overview

This project implements a full quantitative forecasting workflow for Indian equity markets, combining statistical time-series models, volatility diagnostics, and allocation strategies to evaluate short-horizon predictive signals.

The workflow includes:

- ARIMA forecasting
- Prophet forecasting
- Ensemble modelling
- Volatility regime analysis
- Walk-forward validation
- Portfolio optimization
- Residual diagnostics

The primary objective is to assess whether classical forecasting methods can generate economically meaningful signals for portfolio construction while maintaining strict validation discipline and avoiding look-ahead bias.

---

# NSE Equities Analysed

- RELIANCE.NS
- HDFCBANK.NS
- TATASTEEL.NS
- APOLLOHOSP.NS
- MARUTI.NS

---

# Methodology

1. Data acquisition using Yahoo Finance
2. Data preprocessing and stationarity testing
3. ARIMA order selection via AIC grid search
4. Rolling walk-forward forecasting validation
5. Prophet forecasting
6. Inverse-MAPE ensemble construction
7. Volatility regime analysis
8. Forecast-return portfolio allocation
9. Residual diagnostics and model evaluation

---

# Key Features

- Strict walk-forward validation
- Avoidance of look-ahead bias
- ADF stationarity testing
- Ljung-Box residual diagnostics
- Jarque-Bera normality testing
- Rolling volatility analysis
- Forecast-return portfolio construction
- Inverse-volatility weighting

---

# Visual Results

## Dashboard

![Dashboard](outputs/dashboard.png)

## Forecast Confidence Intervals

![Forecasts](outputs/forecast_confidence_intervals.png)

## Residual Diagnostics

![Residuals](outputs/residual_diagnostics.png)

---

# Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Prophet
- Scikit-learn
- Matplotlib
- Seaborn

---

# Repository Structure

```text
forecast-driven-portfolio-allocation/
│
├── notebooks/
├── outputs/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Author

Kiran Kumar  
Indian Institute of Technology Guwahati
