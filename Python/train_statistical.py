# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:53:06 2023

@author: lawre
"""
from train_test_split_custom import feature_label_split, train_val_test_split, train_val_test_split_date, train_val_test_split_idx
from subset import fwd_subset, vif_correlation_subset, corr_cutoff
from predict_features import predict_features
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numba import jit, cuda
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import product
import numpy as np
from error_metrics import smape
from prep_data import prep_data
from statsmodels.tsa.seasonal import seasonal_decompose

# function optimized to run on gpu 
#@jit(target_backend='cuda')   
def train_stat(
            dat,
            dte,
            y_name,
            y_remove,
            step,
            xtra_desc = "",
            run_vif = True,
            run_fwd_selection = True,
            verbose = True,
            read_vif_values = False,
            read_subset_values = False,
            val_days = 50,
            test_days = 50,
            model_name = "ols",
            read_pred_values = False,
            run_corr_cutoff = False,
            read_corr_values = False,            
            max_subset_cols = None,
            rss_cutoff = 5):
    
    
    fcst_y_only = pd.DataFrame()
    model_score_y_only = pd.DataFrame()
    
    #Split data
    #Train test validation splits
    dat = dat.copy().drop(y_remove, axis=1)#.astype('float64')
    
    scaler = MinMaxScaler()
    dat_sc = pd.DataFrame(scaler.fit_transform(dat), index = dat.index, columns=dat.columns)
    dat_sc[y_name] = dat[y_name]
    
    if run_fwd_selection:
        X_train, X_val, X_test_pred, y_train, y_val, y_test, reg_model, subsets = prep_data(dat = dat, y_name = y_name, dte = dte, 
                                                                                       read_pred_values = read_pred_values,
                                                                   val_days = val_days, test_days = test_days, xtra_desc = xtra_desc,
                                                                   run_fwd_selection = run_fwd_selection, read_vif_values = read_vif_values,
                                                                   read_subset_values = read_subset_values, max_subset_cols = max_subset_cols,
                                                                   rss_cutoff = rss_cutoff) 
    else:
        X_train, X_val, X_test_pred, y_train, y_val, y_test = prep_data(dat = dat, y_name = y_name, dte = dte, read_pred_values = read_pred_values,
                                                                   val_days = val_days, test_days = test_days, xtra_desc = xtra_desc,
                                                                   run_fwd_selection = run_fwd_selection, read_vif_values = read_vif_values,
                                                                   read_subset_values = read_subset_values, max_subset_cols = max_subset_cols,
                                                                   rss_cutoff = rss_cutoff) 
    
    y_arima = y_train#X_train_dte[X_train.columns[i]]
    
    result = adfuller(y_arima.values)
    p_value = result[1]
    
    if p_value > 0.05:  # If p-value is greater than 0.05, data is not stationary
        d = 1  # Differencing is needed
        diff_y = y_arima.diff(periods=d).dropna()
    else:
        d = 0  # Data is already stationary
        diff_y = y_arima
    
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
        
    y_arima = y_arima.set_index(pd.to_datetime(y_arima.index)).asfreq('d').ffill()
    X_test_pred_d = X_test_pred.set_index(pd.to_datetime(X_test_pred.index)).asfreq('d').ffill()
    X_train_d = X_train.set_index(pd.to_datetime(X_train.index)).asfreq('d').ffill()
    
    if verbose:
        print("P Value SD Fuller:" + str(round(p_value,3)))
        print("Best Order (p, d, q):", best_order)
        print("Best AIC:", round(best_aic,0))
    
    if model_name == "ols":

        y_pred_train_reg = reg_model.predict(X_train) 
    
        y_pred_test_reg = reg_model.predict(X_test_pred) 
        #y_pred_val_reg = reg_model.predict(X_val) 
    
    elif model_name == "arima_sea":
        
        #https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
        try:
            
            yd = seasonal_decompose(y_arima, model='additive')
            
            model_sea = ARIMA(yd.seasonal, order=best_order)
            model_fit_sea = model_sea.fit()
            model_pred_sea = model_fit_sea.forecast(steps = int(X_test_pred_d.shape[0]), dates = X_test_pred.index)
            
            #trend = yd.trend.fillna(method='ffill').fillna(method='bfill')
            model = ARIMA(y_arima, order=best_order, exog = yd.seasonal)
            model_fit = model.fit()
            model_pred = model_fit.forecast(steps = int(X_test_pred_d.shape[0]), dates = X_test_pred_d.index, exog = model_pred_sea, trend = 'nc')
        
        except:
            
            model = ARIMA(y_arima, order=best_order)
            model_fit = model.fit()
            model_pred = model_fit.forecast(steps = int(X_test_pred_d.shape[0]), dates = X_test_pred_d.index)
            
        y_pred_train_reg = model_fit.fittedvalues[model_fit.fittedvalues.index.isin(X_train.index)]
        y_pred_test_reg = model_pred[model_pred.index.isin(X_test_pred.index)]
        
    elif model_name == "arima_xreg":

            try:
                yd = seasonal_decompose(y_arima, model='additive')
                
                model_sea = ARIMA(yd.seasonal, order=best_order)
                model_fit_sea = model_sea.fit()
                model_pred_sea = model_fit_sea.forecast(steps = int(X_test_pred_d.shape[0]), dates = X_test_pred.index)
                
                #trend = yd.trend.fillna(method='ffill').fillna(method='bfill')
                model = ARIMA(y_arima, order=best_order, exog = pd.merge(yd.seasonal, X_train_d, right_index = True, left_index = True))
                model_fit = model.fit()
                model_pred = model_fit.forecast(steps = int(X_test_pred_d.shape[0]), dates = X_test_pred_d.index, 
                                                exog = pd.merge(model_pred_sea, X_test_pred_d, left_index=True, right_index=True, how = 'right'))
                
            except:
                
                model = ARIMA(y_arima, order=best_order, exog = X_train_d)
                model_fit = model.fit()
                model_pred = model_fit.forecast(steps = int(X_test_pred_d.shape[0]), dates = X_test_pred_d.index, exog = X_test_pred_d)
                
            y_pred_train_reg = model_fit.fittedvalues[model_fit.fittedvalues.index.isin(X_train.index)]
            y_pred_test_reg = model_pred[model_pred.index.isin(X_test_pred.index)]
            
    
    train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_train_reg))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred_test_reg))
    #val_rmse = math.sqrt(mean_squared_error(y_val, y_pred_val_reg))
    
    # calculate smape
    train_smape = smape(y_train.values.flatten(), y_pred_train_reg.values)
    test_smape = smape(y_test.values.flatten(), y_pred_test_reg.values)
    #val_smape = smape(y_val.values.flatten(), y_pred_val_reg.values)
    
    # calculate mase
    train_mase = mean_absolute_percentage_error(y_train, y_pred_train_reg)
    test_mase = mean_absolute_percentage_error(y_test, y_pred_test_reg)
    #val_mase = mean_absolute_percentage_error(y_val, y_pred_val_reg)
    
    
    if verbose == True:
        print('Regression Train Score: %.2f RMSE' % (train_rmse))
        print('Regression Test Score: %.2f RMSE' % (test_rmse))
        #print('Regression Validation Score: %.2f RMSE' % (val_rmse))
        
        plt.plot(subsets["r_sq"], label="r squared")
        plt.legend()
        plt.show()
        
        # Visualising the results
        figure, axes = plt.subplots()
        axes.xaxis_date()

        axes.plot(pd.to_datetime(y_test.index), y_test.values, color = 'red', label = 'Real')
        axes.plot(pd.to_datetime(y_test.index), y_pred_test_reg, color = 'blue', label = 'Predicted')
        plt.title("Regression" + xtra_desc + dte.strftime("%Y%m%d") + ' prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig("output/" + dte.strftime("%Y%m%d") + str(model_name) + xtra_desc + '_prediction.png')
        plt.show()


    model_score = pd.DataFrame(data = {"model_name": model_name, "step": str(step), "dte": str(dte),
                                       "train_rmse": train_rmse, "test_rmse":test_rmse,
                                       "train_smape": train_smape, "test_smape":test_smape,
                                       "train_mase": train_mase, "test_mase":test_mase
                                       }, 
                               index = [model_name + "_" + str(step) + "_" + str(dte)])
    
    fcst = pd.DataFrame(data = {"model_name": str(model_name), "step":[step] * len(y_test), 
                                "dte" : y_test.index.to_list(), 
                                "y_pred": y_pred_test_reg.values.flatten(), "y_obs": y_test.values.flatten()})
    
    fcst_y_only = pd.concat([fcst_y_only, fcst], axis = 0)
    model_score_y_only = pd.concat([model_score_y_only, model_score], axis = 0)
    

    return(fcst_y_only, model_score_y_only)
