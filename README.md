# Master_thesis
 Predicting the accuracy of cryptocurrency volatility with ARMA-GARCH, Prophet and LSTM models

## Data description
Using data: BTC,ETH,BNB  
Data sources: Binance API  
Proxy Variable: Realized Volatility(RV).  
 **-> In this study I bulit the day of RV with every 3min,5min and 15min <-**  
Number of data： 879,952, 529,153 and 176,783
Date: 2017/11/06~2022/11/27  
  in-sample observations:2017/11/06 ~ 2021/05/21(1,293 days)  
  out-of-sample observations:2021/05/22 ~ 2022/11/27(555 days)  
Evaluation indicators: MAE,RMSE,MAPE  
**-> Using sliding window algorithm <-**  
## Data Preprocessing
 1. Translating to every day of RV
 2. (ARMA-GARCH)Making sure data is stable by ADF
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
