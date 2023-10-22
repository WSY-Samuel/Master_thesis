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
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import statsmodels.api as sm
import pmdarima as pmd
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'

start = time.time() # 開始測量

df_btc = pd.read_excel('/Users/wangshuyou/碩論資料/btc_5min.xlsx',index_col='date')
df_btc = df_btc.sort_index()
                   
df_btc['close_lag'] = df_btc['close'].shift(1)
df_btc['ln_clo'] = df_btc['close'].apply(np.log)
df_btc['ln_clo_lag'] = df_btc['close_lag'].apply(np.log)
df_btc['r'] = df_btc['ln_clo'] - df_btc['ln_clo_lag']
df_btc = df_btc.where(df_btc.notnull(),0)


# Day train/test set 73分
train = df_btc.loc[:'2021-05-21 23:55:00',:]
test = pd.concat([df_btc,train]).drop_duplicates(keep = False)

# 日收益率
def rv(x):
    return np.sum(x)

r_all = df_btc['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
r_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
r_test = test['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)

# 日波動
def sd(x):
    sd = np.sqrt(np.sum(x**2))
    return sd

sd_all = df_btc['r'].groupby(pd.Grouper(freq = 'D')).apply(sd)
sd_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(sd)
test_sd = test['r'].groupby(pd.Grouper(freq = 'D')).apply(sd)

# =============================================================================
# # ADF檢定
# def adf_test(df):
#     dftest = ADF(df)
#     dfoutput = pd.Series(dftest[0:4],index=['ADF-Statistic','P-Value','Lag','Number'])
#     for key,value in dftest[4].items(): #取得字典的每對 key & value
#         dfoutput['Critical Value(%s)' %key] = value
#     print(dfoutput)
#     if dfoutput['P-Value'] > 0.05:
#         print('不穩定')
#     else:
#         print('穩定')
# adf_test(df_btc['r']) # pv = 0.000 穩定
# =============================================================================

# =============================================================================
# # acf,pacf(收益率)
# fig,ax = plt.subplots()
# fig = plot_acf(r_train)
# fig = plot_pacf(r_train)
# plt.show()
# 
# =============================================================================

# =============================================================================
# #白噪音檢定
# print('White_Noise Test:', acorr_ljungbox(r_train)) #返回统计量和p值
# =============================================================================
# lag1_pv : 3.025153e-143\lag2_pv : 2.718971e-218\lag3_pv : 4.712549e-283 皆小於0.05，為非白噪音
# =============================================================================
# 
# # AIC fun.1
# from statsmodels.tsa.arima.model import ARIMA
# import itertools
# p = range(0,2) # 0~5
# q = range(0,2) # 0~5
# d = range(0,1) # 0
# pdq = list(itertools.product(p,d,q)) # 列出242個個數
# 
# aic_value = {} #創建一個空的字典
# 
# for param in pdq:
#     try:
#         model_arima = ARIMA(r_train,order=param)
#         model_arima_fit = model_arima.fit()
#         print(param,model_arima_fit.aic)
#         aic_value.setdefault(param,model_arima_fit.aic)
#     except:
#         continue
# print("AIC最小組合：",min(aic_value, key = aic_value.get))
# =============================================================================
# =============================================================================
# # AIC最小組合： (3, 1)
# pmd_mdl = pmd.auto_arima(r_train)
# print(pmd_mdl.summary())
# =============================================================================

# ARMA模型建置
model = sm.tsa.ARIMA(r_train, order = (0,0,1))
arma = model.fit() 
print(arma.summary())
# 取得ARMA模型的殘差項目
arma_resid = list(arma.resid)
#sm.tsa.stattools.arma_order_select_ic(sd_train,max_ar = 10, max_ma = 10, ic = ['aic'])
# AIC最小組合： (2, 3)
# GARCH模型建置
mdl_garch = arch_model(arma_resid, vol = 'GARCH', p = 1, q = 1)
garch = mdl_garch.fit()
print(garch.summary())


# sliding window
y_pred = pd.DataFrame()
right = 1293
for i in range(555):
    train = r_all.iloc[i:right]
    # 定義、訓練模型
    garch_model = arch_model(train,vol = 'GARCH', p=1, q=1, dist='normal')
    # vol: 波動度模型
    gm_result = garch_model.fit(disp='off')
    # iter:最佳化的頻率  disp:返回收斂值
    # 建構預測集
    gm_forecast = gm_result.forecast(reindex = False,horizon=1).variance 
    # reindex = True: 使用未來的數據 reindex = False:包含過去
    y_pred = y_pred.append(np.sqrt(gm_forecast))
    right += 1
    

y_true = pd.DataFrame(test_sd)
y_true = y_true.reset_index()
y_true['date'] = pd.to_datetime(y_true['date'])
y_true['r'] = y_true['r'].apply(lambda x: float(x))
# 解決： TypeError: cannot convert the series to <class 'float'> 問題


y_pred['h.1'] = y_pred['h.1'].apply(lambda x: float(x))
y_pred = y_pred.reset_index()
y_pred['date'] = pd.to_datetime(y_pred['date'])
# =============================================================================
# 解決： TypeError: cannot convert the series to <class 'float'> 問題

# plot expected vs actual

plt.plot(y_true['date'], y_true['r'], label='實際值', linewidth = 0.7)
plt.plot(y_pred['date'], y_pred['h.1'], label='預測值',c = 'orange')
#plt.title('ARMA-Garch(1,1)_RV_predict_1D')
plt.xlabel('日期')
plt.ylabel('實際波動度（RV）')
plt.xticks(rotation = 45)
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

print('MAE:%.4f ' % mae(y_true['r'],y_pred['h.1']))
print('RMSE:%.4f ' % rmse(y_true['r'],y_pred['h.1']))
print('MAPE:%.4f' % mape(y_true['r'],y_pred['h.1']))


end = time.time() # 結束測量
print("執行時間：%f 秒" % (end - start))

##############################################################################
df_eth = pd.read_excel('/Users/wangshuyou/碩論資料/eth_15min.xlsx',index_col='date')
df_eth = df_eth.sort_index()
                   
df_eth['close_lag'] = df_eth['close'].shift(1)
df_eth['ln_clo'] = df_eth['close'].apply(np.log)
df_eth['ln_clo_lag'] = df_eth['close_lag'].apply(np.log)
df_eth['r'] = df_eth['ln_clo'] - df_eth['ln_clo_lag']
df_eth = df_eth.where(df_eth.notnull(),0)


# Day train/test set 73分
train = df_eth.loc[:'2021-05-21 23:45:00',:]
test = pd.concat([df_eth,train]).drop_duplicates(keep = False)

# 日收益率
def rv(x):
    return np.sum(x)
# 日收益率
r_all = df_eth['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
r_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
r_test = test['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)

def sd(x):
    sd = np.sqrt(np.sum(x**2))
    return sd
# 日波動
sd_all = df_eth['r'].groupby(pd.Grouper(freq = 'D')).apply(sd)
sd_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(sd)
test_sd = test['r'].groupby(pd.Grouper(freq = 'D')).apply(sd)

# =============================================================================
# def adf_test(df):
#     dftest = ADF(df)
#     dfoutput = pd.Series(dftest[0:4],index=['ADF-Statistic','P-Value','Lag','Number'])
#     for key,value in dftest[4].items(): #取得字典的每對 key & value
#         dfoutput['Critical Value(%s)' %key] = value
#     print(dfoutput)
#     if dfoutput['P-Value'] > 0.05:
#         print('不穩定')
#     else:
#         print('穩定')
# adf_test(df_eth['r']) # pv = 0.000 穩定
# =============================================================================

# ARMA模型建置
eth_model = sm.tsa.ARIMA(r_train, order = (0,0,1))
arma = eth_model.fit() 
print(arma.summary())
# 取得ARMA模型的殘差項目
arma_resid = list(arma.resid)
#sm.tsa.stattools.arma_order_select_ic(sd_train,max_ar = 10, max_ma = 10, ic = ['aic'])

# GARCH模型建置
mdl_garch = arch_model(arma_resid, vol = 'GARCH', p = 1, q = 1)
garch = mdl_garch.fit()
print(garch.summary())


# sliding window
y_pred = pd.DataFrame()
right = 1293
for i in range(555):
    train = r_all.iloc[i:right]
    # 定義、訓練模型
    garch_model = arch_model(train,vol = 'GARCH', p=1, q=1, dist='normal')
    # vol: 波動度模型
    gm_result = garch_model.fit(disp='off')
    # iter:最佳化的頻率  disp:返回收斂值
    # 建構預測集
    gm_forecast = gm_result.forecast(reindex = False,horizon=1).variance 
    # reindex = True: 使用未來的數據 reindex = False:包含過去
    y_pred = y_pred.append(np.sqrt(gm_forecast))
    right += 1
    

y_true = pd.DataFrame(test_sd)
y_true = y_true.reset_index()
y_true['date'] = pd.to_datetime(y_true['date'])
y_true['r'] = y_true['r'].apply(lambda x: float(x))
# 解決： TypeError: cannot convert the series to <class 'float'> 問題


y_pred['h.1'] = y_pred['h.1'].apply(lambda x: float(x))
y_pred = y_pred.reset_index()
y_pred['date'] = pd.to_datetime(y_pred['date'])
# =============================================================================
# 解決： TypeError: cannot convert the series to <class 'float'> 問題

# plot expected vs actual
plt.plot(y_true['date'], y_true['r'], label='實際值', linewidth = 0.7)
plt.plot(y_pred['date'], y_pred['h.1'], label='預測值',c = 'orange')
#plt.title('ARMA-Garch(1,1)_RV_predict_1D')
plt.xlabel('日期')
plt.ylabel('實際波動度（RV）')
plt.xticks(rotation = 45)
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


print('MAE:%.4f ' % mae(y_true['r'],y_pred['h.1']))
print('RMSE:%.4f ' % rmse(y_true['r'],y_pred['h.1']))
print('MAPE:%.4f' % mape(y_true['r'],y_pred['h.1']))



end = time.time() # 結束測量
print("執行時間：%f 秒" % (end - start))










