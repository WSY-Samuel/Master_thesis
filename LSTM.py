#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:31:23 2023

@author: wangshuyou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:10:19 2022

@author: wangshuyou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation
from keras.layers.core import Flatten
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import time
import random
import warnings
warnings.filterwarnings('ignore')

start = time.time() # 開始測量

# 載入訓練資料
df_eth = pd.read_excel("C:\碩論資料\eth.xlsx",index_col='date')
df_eth = df_eth.sort_index()

df_eth['close_lag'] = df_eth['close'].shift(1)
df_eth['ln_clo'] = df_eth['close'].apply(np.log)
df_eth['ln_clo_lag'] = df_eth['close_lag'].apply(np.log)
df_eth['r'] = df_eth['ln_clo'] - df_eth['ln_clo_lag']
df_eth = df_eth.where(df_eth.notnull(),0)


# train/test set 73分
train = df_eth.loc[:'2021-04-27 23:55:00',:]
test = pd.concat([df_eth,train]).drop_duplicates(keep = False) 

# 實際波動度
def rv(x):
    return np.sqrt(np.sum(x**2))
rv_all = df_eth['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
rv_train = train['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)
rv_test = test['r'].groupby(pd.Grouper(freq = 'D')).apply(rv)

# 1349筆資料預測下1期
xx,y = [],[]
for i  in range(1349,len(rv_all)):
    xx.append(rv_all.iloc[i-1349:i])
    y.append(rv_all.iloc[i])
xx,y = np.array(xx),np.array(y) # array特性,行列相反
xx = np.reshape(xx, (xx.shape[0],xx.shape[1],1)) #轉成3維,INDEX=0 是前1349個
# 3維的意義:[樣本\時間步\功能]

#Build the model
# 共五層:3 hidden layers + 2 fully connected layers
random.seed(0)
model = Sequential()
# 第1層layers
model.add(LSTM(10,input_shape=(xx.shape[1],1),return_sequences=True))
model.add(Dropout(0.3))# 捨棄率,防止overfitting 
# units = 神經元的數目 
# 第2層layers
model.add(LSTM(4, return_sequences=True))
model.add(Dropout(0.8)) 
# 第3層layers
model.add(LSTM(2, return_sequences=True))
model.add(Dropout(0.8)) 
model.add(Dense(1))
model.add(Flatten())
# 第4層connected layer
model.add(Dense(5))
# 第5層connected layer
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(xx, y,epochs=150, batch_size = 1349,verbose=2)
# verbose = 2 :為每個epochs輸出一行紀錄
# epoch : 期，演算法完整使用150次資料集每筆資料的狀態

Xt = model.predict(xx)
plt.plot(y,label = 'True') # 維度更改成1行1列: -1的功能是自動計算，C = 1, C/D(個數) = 576
plt.plot(Xt,label = 'Predict')
plt.xlabel('Days')
plt.ylabel('Realized Volatility')
plt.title('LSTM_RV_predict_1D')
plt.legend()
plt.show()

y = y.reshape(-1,1)
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
# =============================================================================

print('MAE:%3f ' % mae(y,Xt))
print('RMSE:%3f ' % rmse(y,Xt))
print('MAPE:%3f' % mape(y,Xt))


end = time.time() # 結束測量
print("執行時間：%f 秒" % (end - start))





