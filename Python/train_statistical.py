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
            read_pred_values = False,
            run_corr_cutoff = False,
            read_corr_values = False):
    
    
    fcst_y_only = pd.DataFrame()
    model_score_y_only = pd.DataFrame()
    
    #Split data
    #Train test validation splits
    dat = dat.copy().drop(y_remove, axis=1)#.astype('float64')
    
    scaler = MinMaxScaler()
    dat_sc = pd.DataFrame(scaler.fit_transform(dat), index = dat.index, columns=dat.columns)
    dat_sc[y_name] = dat[y_name]
    
    X_train, X_val, X_test_pred, y_train, y_val, y_test, reg_model, subsets = prep_data(dat = dat, y_name = y_name, dte = dte, 
                                                                                        read_pred_values = read_pred_values,
                                                               val_days = val_days, test_days = val_days, xtra_desc = xtra_desc)
    
# =============================================================================
#     X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_idx(df = dat, target_col = y_name, dte = dte,
#                                                                               val_days = val_days, test_days = test_days)
#     
#     #Subset data 
#     if run_corr_cutoff:
#         X_train = corr_cutoff(X_train, y_train, read_corr_values = read_corr_values, verbose = verbose, dte = dte, 
#                               xtra_desc = xtra_desc)
#         
#         X_test = X_test[X_train.columns]        
#         X_val = X_val[X_train.columns] 
#     
#     if run_vif:
#         X_train = vif_correlation_subset(X_train, read_vif_values = read_vif_values, verbose = verbose, dte = dte, 
#                               xtra_desc = xtra_desc)
#         
#         X_test = X_test[X_train.columns]        
#         X_val = X_val[X_train.columns] 
# 
#     if run_fwd_selection:
#         X_train, reg_model, subsets = fwd_subset(X_train, y_train, y_name, verbose = verbose, dte = dte,
#                                                  read_subset_values = read_subset_values, xtra_desc = xtra_desc)
#         
#         X_test = X_test[X_train.columns]
#         X_val = X_val[X_train.columns]
# 
# 
#     if read_pred_values:
#         X_test_pred = pd.read_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_X_test_pred.csv", index_col = 0)
# 
#     else:
#         X_test_pred = predict_features(X_train, X_test, test_days = test_days, verbose = verbose)
#         X_test_pred.to_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_X_test_pred.csv")
#         
#     if verbose == True:
#         print(dte)
#         print("X_train: " + str(X_train.shape) + " " + str(min(X_train.index)) + ">" + str(max(X_train.index)))
#         print("X_test: " + str(X_test.shape)+ " " + str(min(X_test.index)) + ">" + str(max(X_test.index)))
#         print("X_val: " + str(X_val.shape)+ " " + str(min(X_val.index)) + ">" + str(max(X_val.index)))
#         print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
#         print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))
#         print("y_val: " + str(y_val.shape)+ " " + str(min(y_val.index)) + ">" + str(max(y_val.index)))
# 
# =============================================================================
    model_name = "reg"
    y_pred_train_reg = reg_model.predict(X_train) 

    y_pred_test_reg = reg_model.predict(X_test_pred) 
    y_pred_val_reg = reg_model.predict(X_val) 
    
    train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_train_reg))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred_test_reg))
    val_rmse = math.sqrt(mean_squared_error(y_val, y_pred_val_reg))
    
    # calculate smape
    train_smape = smape(y_train, y_train)
    test_smape = smape(y_test, y_test)
    val_smape = smape(y_val, y_val)
    
    # calculate mase
    train_mase = mean_absolute_percentage_error(y_train, y_train)
    test_mase = mean_absolute_percentage_error(y_test, y_test)
    val_mase = mean_absolute_percentage_error(y_val, y_val)
    
    
    if verbose == True:
        print('Regression Train Score: %.2f RMSE' % (train_rmse))
        print('Regression Test Score: %.2f RMSE' % (test_rmse))
        print('Regression Validation Score: %.2f RMSE' % (val_rmse))
        
        plt.plot(subsets["r_sq"], label="r squared")
        plt.legend()
        plt.show()
        
        # Visualising the results
        figure, axes = plt.subplots()
        axes.xaxis_date()

        axes.plot(pd.to_datetime(y_test.index), y_test.values, color = 'red', label = 'Real')
        axes.plot(pd.to_datetime(y_test.index), y_pred_test_reg, color = 'blue', label = 'Predicted')
        plt.title("Regression" + xtra_desc + str(dte) + ' prediction')
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
