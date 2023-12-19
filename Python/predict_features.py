# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:40:36 2023

@author: lawre
"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import product
import numpy as np
import math
from numba import jit, cuda
from statsmodels.tsa.seasonal import seasonal_decompose

jit(target_backend='cuda') 
def predict_features(X_train, X_test, test_days, verbose = True):
    X_test_pred = pd.DataFrame(index = X_test.index, columns = X_test.columns)
    
    X_train_dte = X_train.set_index(pd.to_datetime(X_train.index)).asfreq('d').ffill()
    
    for i in range(0,len(X_test.columns)):   
        
        y = X_train_dte[X_train.columns[i]]
        
        result = adfuller(y.values)
        p_value = result[1]
        
        if p_value > 0.05:  # If p-value is greater than 0.05, data is not stationary
            d = 1  # Differencing is needed
            diff_y = y.diff(periods=d).dropna()
        else:
            d = 0  # Data is already stationary
            diff_y = y
        
        p = range(0, 3)  # Choose a range for p
        d = range(0, 2)  # Choose a range for d
        q = range(0, 3)  # Choose a range for q
        
        best_aic = np.inf
        best_order = None
        
        for order in product(p, d, q):
            try:
                model = sm.tsa.ARIMA(diff_y, order=order)
                results = model.fit()
                aic = results.aic
        
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
            except:
                continue
        
        if verbose:
            print(X_train.columns[i])
            print("P Value SD Fuller:" + str(round(p_value,3)))
            print("Best Order (p, d, q):", best_order)
            print("Best AIC:", round(best_aic,0))
        
        #https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
    
        yd = seasonal_decompose(y, model='additive')
        
        model_sea = ARIMA(yd.seasonal, order=best_order)
        model_fit_sea = model_sea.fit()
        model_pred_sea = model_fit_sea.forecast(steps = int(test_days), dates = X_test.index)
        
        #trend = yd.trend.fillna(method='ffill').fillna(method='bfill')
        model = ARIMA(y, order=best_order, exog = yd.seasonal)
        model_fit = model.fit()
        model_pred = model_fit.forecast(steps = int(test_days), dates = X_test.index, exog = model_pred_sea, trend = 'nc')
        
        X_test_pred[X_test_pred.columns[i]] = model_pred.values
    
    return(X_test_pred)