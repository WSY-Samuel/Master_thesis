# Master_thesis
 Predicting the accuracy of cryptocurrency volatility with ARMA-GARCH, Prophet and LSTM models

## Data description
1. **Data Used:**
   - Cryptocurrencies: BTC (Bitcoin), ETH (Ethereum), BNB (Binance Coin).
   - Data Sources: Obtained from the Binance API.

2. **Proxy Variable:**
   - Realized Volatility (RV).
   - Building the daily RV at intervals of every 3 minutes, 5 minutes, and 15 minutes.

3. **Number of Data Points:**
   - 879,952 data points for one frequency.(3 minutes)
   - 529,153 data points for another frequency.(5 minutes)
   - 176,783 data points for yet another frequency.(15 minutes)

4. **Time Period:**
   - Date Range: November 6, 2017, to November 27, 2022.
   - In-sample observations cover November 6, 2017, to May 21, 2021 (1,293 days).
   - Out-of-sample observations cover May 22, 2021, to November 27, 2022 (555 days).

5. **Evaluation Indicators:**
   - Mean Absolute Error (MAE).
   - Root Mean Squared Error (RMSE).
   - Mean Absolute Percentage Error (MAPE).
## Data Preprocessing
1. **Translating to Daily Realized Volatility (RV):**
   - The process involves translating the data to daily Realized Volatility (RV).

2. **Stability Check using ARMA-GARCH:**
   - Employing Autoregressive Integrated Moving Average (ARMA) and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models for stability verification.
   - Conducting Augmented Dickey-Fuller (ADF) tests and examining Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to ensure data stability.

3. **Lag Selection for ARMA using AIC:**
   - Determining the lag order for the Autoregressive Moving Average (ARMA) model based on the Akaike Information Criterion (AIC).
## Model Building
### ARMA(0,1)-GARCH(1,1)
### Prophet
### LSTM  
1. Architecture:
   - Three layers of LSTM.
   - Two fully connected layers.

2. Hyperparameters:
   - Learning Rate: 0.0001
   - Neurons: (10, 4, 2)
   - Dropout Rates: (0.3, 0.8, 0.8)

### Conclusion:
In summary, the LSTM model demonstrates superior performance. The accuracy is notably better compared to the other two models. Conversely, the Prophet model exhibits the least favorable performance in this scenario.

