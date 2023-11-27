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
 1. Translating to every day of RV
 2. (ARMA-GARCH)Making sure data is stable by ADF and ACF,PACF
 3. Deciding lag of ARMA by AIC
## Model Building
### ARMA(0,1)-GARCH(1,1)
### Prophet
### LSTM  
 1. 3 layers of LSTM and 2 layers of connettion
 2. learning rate:0.0001
 3. Neurons：(10, 4, 2)
 4. Dropout rating:(0.3, 0.8, 0.8)
## Concludion
Overall,the LSTM model performs best, and it can be clearly seen that the accuracy is significantly better than the other two. The Prophet model performed the worst this time around.
