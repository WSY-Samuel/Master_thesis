#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:25:26 2022

@author: wangshuyou
"""
import pandas as pd 
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'

start = time.time() # 開始測量

df_eth = pd.read_excel('/Users/wangshuyou/碩論資料/eth_5min.xlsx',index_col='date')
df_eth = df_eth.sort_index()
#df_eth_d = pd.read_excel('/Users/wangshuyou/碩論資料/eth_1day_原.xlsx',index_col='date')
#df_eth_d['return'] = df_eth_d['close'].pct_change()
# 報酬率敘述統計
#summary = df_eth_d['return'].describe()

# =============================================================================
# # 1min轉乘5min
# df = []
# for i in range(0,len(df_e),5):
#     df.append(i)
# df_eth = df_e.iloc[df]
# =============================================================================

# =============================================================================
# # Daily closing prices
# sns.lineplot(x = df_eth_d.index, y = 'close', data = df_eth_d)
# plt.title('Daily close prices of ETH')
# plt.show()
# 
# #Daily return of ETH
# df_eth_d['returns'] = df_eth_d['close'].pct_change()
# sns.lineplot(x = df_eth_d.index, y = 'return', data = df_eth_d,sizes=0.3)
# plt.title('Daily return of ETH')
# plt.show()
# =============================================================================
                   
df_eth['close_lag'] = df_eth['close'].shift(1)
df_eth['ln_clo'] = df_eth['close'].apply(np.log)
df_eth['ln_clo_lag'] = df_eth['close_lag'].apply(np.log)
df_eth['r'] = df_eth['ln_clo'] - df_eth['ln_clo_lag']
df_eth = df_eth.where(df_eth.notnull(),0)

#intraday return of ETH
#sns.lineplot(x = df_eth.index, y = 'r', data = df_eth,sizes=0.3)
#plt.title('intraday return of ETH')
#plt.show()

#s = df_eth['r'].describe()
# =============================================================================
# # Day train/test set 73分
# train_d = df_eth_d.iloc[:math.floor(0.7*len(df_eth_d)),:]
# #train = train.set_index('date')
# test_d = pd.concat([df_eth_d,train_d]).drop_duplicates(keep=False) #差集  
# =============================================================================

train = df_eth.loc[:'2021-05-21 23:55:00',:]
test = pd.concat([df_eth,train]).drop_duplicates(keep = False)

# 實際波動度：對數收益率的標準差
def rv(x):
    return np.sqrt(np.sum(x**2))

rv_all = df_eth['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
rv_all = rv_all.reset_index()
rv_all = rv_all.set_axis(['ds','y'], axis =1)

rv_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
rv_train = rv_train.reset_index()
rv_train = rv_train.set_axis(['ds','y'], axis =1)
print(rv_train.describe())
test_rv = test['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)

# sliding window 
y_pred = []
right = 1293 
for i in range(555):
    train = rv_all.iloc[i:right]
    # 定義、訓練模型
    model = Prophet()
    model.fit(train)
    # 建構預測集
    future = model.make_future_dataframe(periods=1 ,freq='D')
    # 進行預測
    forecast = model.predict(future)
    y_pred.append(forecast['yhat'].iloc[-1])
    #print(train['ds'].head(1))
    #print(train['ds'].tail(1))
    #print('======================')
    right += 1
y_pred = pd.DataFrame(y_pred)
y_pred = y_pred.set_axis(['r'],axis = 1)
y_pred['r'] = y_pred['r'].apply(lambda x: float(x))
y_pred['date'] = test_rv.index
y_pred['date'] = pd.to_datetime(y_pred['date'])

# train set 
train_set = rv_all.iloc[0:1349]
model1 = Prophet()
model1.fit(train_set)
future_train = model1.make_future_dataframe(periods=1 ,freq='D')
forecast_train = model1.predict(future_train)
figure=model1.plot(forecast_train)
plt.show()
# daily seasonality
fig = model1.plot_components(forecast_train)
plt.show()

#y_true = y_true['r']
y_true = pd.DataFrame(test_rv)
y_true = y_true.reset_index()
y_true['date'] = pd.to_datetime(y_true['date'])
y_true['r'] = y_true['r'].apply(lambda x: float(x))
# =============================================================================
# y_true = pd.DataFrame(y_true)
# y_true_time = test_rv.index.strftime('%Y/%m/%d')
# y_true = y_true.set_index(y_true_time)
# =============================================================================


# plot expected vs actual
plt.plot(y_true['date'],y_true['r'], label='實際值', linewidth = 0.7)
plt.plot(y_pred['date'],y_pred['r'], label='預測值',c = 'orange')
#plt.title('Prophet_RV_predict_1D')
plt.xticks(rotation = 45)
plt.xlabel('日期')
plt.ylabel('實際波動度（RV）')
plt.legend()
plt.show()

# MAE
def mae(y,yhat):
    mae = sum(abs(y-yhat))/len(y)
    return mae
#MSE 可以評價資料的變化程度
def rmse(y, yhat):
    rmse = np.sqrt(sum(((y - yhat)*(y - yhat))))
    return rmse
#MAPE：表示預測值和實際值之間的平均偏差為？%
def mape(y, yhat):
    mape = (y - yhat)/y
    mape[~np.isfinite(mape)] = 0 # 先計算除法，在處理除以0變成inf的問題
    mape = np.mean(np.abs(mape)) * 100
    return mape

print('MAE:%.3f ' % mae(y_true['r'],y_pred['r']))
print('RMSE:%.3f ' % rmse(y_true['r'],y_pred['r']))
print('MAPE:%.3f' % mape(y_true['r'],y_pred['r']))

end = time.time() # 結束測量
print("執行時間：%f 秒" % (end - start))

######################################################################
start = time.time() # 開始測量

df_eth = pd.read_excel('/Users/wangshuyou/碩論資料/eth_3min.xlsx',index_col='date')
df_eth = df_eth.sort_index()
#df_eth_d = pd.read_excel('/Users/wangshuyou/碩論資料/eth_1day_原.xlsx',index_col='date')
#df_eth_d['return'] = df_eth_d['close'].pct_change()
# 報酬率敘述統計
#summary = df_eth_d['return'].describe()

# =============================================================================
# # 1min轉乘5min
# df = []
# for i in range(0,len(df_e),5):
#     df.append(i)
# df_eth = df_e.iloc[df]
# =============================================================================

# =============================================================================
# # Daily closing prices
# sns.lineplot(x = df_eth_d.index, y = 'close', data = df_eth_d)
# plt.title('Daily close prices of ETH')
# plt.show()
# 
# #Daily return of ETH
# df_eth_d['returns'] = df_eth_d['close'].pct_change()
# sns.lineplot(x = df_eth_d.index, y = 'return', data = df_eth_d,sizes=0.3)
# plt.title('Daily return of ETH')
# plt.show()
# =============================================================================
                   
df_eth['close_lag'] = df_eth['close'].shift(1)
df_eth['ln_clo'] = df_eth['close'].apply(np.log)
df_eth['ln_clo_lag'] = df_eth['close_lag'].apply(np.log)
df_eth['r'] = df_eth['ln_clo'] - df_eth['ln_clo_lag']
df_eth = df_eth.where(df_eth.notnull(),0)

#intraday return of ETH
#sns.lineplot(x = df_eth.index, y = 'r', data = df_eth,sizes=0.3)
#plt.title('intraday return of ETH')
#plt.show()

#s = df_eth['r'].describe()
# =============================================================================
# # Day train/test set 73分
# train_d = df_eth_d.iloc[:math.floor(0.7*len(df_eth_d)),:]
# #train = train.set_index('date')
# test_d = pd.concat([df_eth_d,train_d]).drop_duplicates(keep=False) #差集  
# =============================================================================

train = df_eth.loc[:'2021-05-21 23:57:00',:]
test = pd.concat([df_eth,train]).drop_duplicates(keep = False)

# 實際波動度：對數收益率的標準差
def rv(x):
    return np.sqrt(np.sum(x**2))

rv_all = df_eth['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
rv_all = rv_all.reset_index()
rv_all = rv_all.set_axis(['ds','y'], axis =1)

rv_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
rv_train = rv_train.reset_index()
rv_train = rv_train.set_axis(['ds','y'], axis =1)
print(rv_train.describe())
test_rv = test['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)

# sliding window 
y_pred = []
right = 1293 
for i in range(555):
    train = rv_all.iloc[i:right]
    # 定義、訓練模型
    model = Prophet()
    model.fit(train)
    # 建構預測集
    future = model.make_future_dataframe(periods=1 ,freq='D')
    # 進行預測
    forecast = model.predict(future)
    y_pred.append(forecast['yhat'].iloc[-1])
    #print(train['ds'].head(1))
    #print(train['ds'].tail(1))
    #print('======================')
    right += 1
y_pred = pd.DataFrame(y_pred)
y_pred = y_pred.set_axis(['r'],axis = 1)
y_pred['r'] = y_pred['r'].apply(lambda x: float(x))
y_pred['date'] = test_rv.index
y_pred['date'] = pd.to_datetime(y_pred['date'])

# train set 
train_set = rv_all.iloc[0:1349]
model1 = Prophet()
model1.fit(train_set)
future_train = model1.make_future_dataframe(periods=1 ,freq='D')
forecast_train = model1.predict(future_train)
figure=model1.plot(forecast_train)
plt.show()
# daily seasonality
fig = model1.plot_components(forecast_train)
plt.show()

#y_true = y_true['r']
y_true = pd.DataFrame(test_rv)
y_true = y_true.reset_index()
y_true['date'] = pd.to_datetime(y_true['date'])
y_true['r'] = y_true['r'].apply(lambda x: float(x))
# =============================================================================
# y_true = pd.DataFrame(y_true)
# y_true_time = test_rv.index.strftime('%Y/%m/%d')
# y_true = y_true.set_index(y_true_time)
# =============================================================================


# plot expected vs actual
plt.plot(y_true['date'],y_true['r'], label='實際值', linewidth = 0.7)
plt.plot(y_pred['date'],y_pred['r'], label='預測值',c = 'orange')
#plt.title('Prophet_RV_predict_1D')
plt.xticks(rotation = 45)
plt.xlabel('日期')
plt.ylabel('實際波動度（RV）')
plt.legend()
plt.show()

# MAE
def mae(y,yhat):
    mae = sum(abs(y-yhat))/len(y)
    return mae
#MSE 可以評價資料的變化程度
def rmse(y, yhat):
    rmse = np.sqrt(sum(((y - yhat)*(y - yhat))))
    return rmse
#MAPE：表示預測值和實際值之間的平均偏差為？%
def mape(y, yhat):
    mape = (y - yhat)/y
    mape[~np.isfinite(mape)] = 0 # 先計算除法，在處理除以0變成inf的問題
    mape = np.mean(np.abs(mape)) * 100
    return mape

print('MAE:%.3f ' % mae(y_true['r'],y_pred['r']))
print('RMSE:%.3f ' % rmse(y_true['r'],y_pred['r']))
print('MAPE:%.3f' % mape(y_true['r'],y_pred['r']))

end = time.time() # 結束測量
print("執行時間：%f 秒" % (end - start))