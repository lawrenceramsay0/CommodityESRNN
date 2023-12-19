# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 23:04:57 2023

@author: lawre
"""
from train_test_split_custom import feature_label_split, train_val_test_split, train_val_test_split_date, train_val_test_split_idx
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from itertools import product
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from error_metrics import smape

def train_y_only(
            dat,
            dte,
            y_name,
            step,
            verbose = True,
            val_days = 50,
            test_days = 50,
            period = "D",
            xtra_desc = ""):
    
    fcst_y_only = pd.DataFrame()#columns = ["model_name", "step", "dte", "y_pred", "y_obs"])
    model_score_y_only = pd.DataFrame()#columns = ["model_name", "step", "dte", "train_rmse", "test_rmse"])
    
    #Train test validation splits
    dat2 = dat.copy()
    dat2.index = pd.DatetimeIndex(dat.index).to_period(period)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_idx(df = dat, target_col = y_name, dte = dte,
                                                                              val_days = val_days, test_days = test_days)
    
    if verbose == True:
        print(dte)
        print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
        print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))

    #model_type = "es"
    #https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html#statsmodels.tsa.holtwinters.ExponentialSmoothing
    es1 = SimpleExpSmoothing(y_train, initialization_method="heuristic").fit()
    fcst1_test = es1.forecast(len(y_test)).rename("SES_alpha=0.2")
    fcst1_train = es1.forecast(len(y_train)).rename("SES_alpha=0.2")
    
    es2 = SimpleExpSmoothing(y_train, initialization_method="heuristic").fit(smoothing_level=0.6, optimized=False)
    fcst2_test = es2.forecast(len(y_test)).rename("SES_alpha=0.6")
    fcst2_train = es2.forecast(len(y_train)).rename("SES_alpha=0.6")
    
    es3 = SimpleExpSmoothing(y_train, initialization_method="estimated").fit()
    fcst3_test = es3.forecast(len(y_test)).rename(r"SES_alpha=%s" % round(es3.model.params["smoothing_level"],2))
    fcst3_train = es3.forecast(len(y_train)).rename(r"SES_alpha=%s" % round(es3.model.params["smoothing_level"],2))

    es4 = ExponentialSmoothing(y_train, initialization_method="estimated").fit()
    fcst4_test = es4.forecast(len(y_test)).rename(r"ES_alpha=%s" % round(es4.model.params["smoothing_level"],2))
    fcst4_train = es4.forecast(len(y_train)).rename(r"ES_alpha=%s" % round(es4.model.params["smoothing_level"],2))
    
    #model_type = "naive"
    fcst5_test = y_test.copy()
    fcst5_test[y_name] = y_train.iloc[-1][y_name]
    
    fcst5_train = y_train.copy()
    fcst5_train[y_name] = y_train.iloc[-1][y_name]
    
    fcst5_test = fcst5_test.rename(columns={y_name: "naive"})
    fcst5_train = fcst5_train.rename(columns={y_name: "naive"})
    
    #model_type = "ma30"
    fcst6_test = y_test.copy()
    fcst6_test[y_name] = y_train.rolling(30).mean().iloc[-1][y_name]
    
    fcst6_train = y_train.copy()
    fcst6_train[y_name] = y_train.rolling(30).mean().iloc[-1][y_name]
    
    fcst6_test = fcst6_test.rename(columns={y_name: "ma30"})
    fcst6_train = fcst6_train.rename(columns={y_name: "ma30"})
    
    #model_type = "holt"
    #https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.Holt.html
    holt1 = Holt(y_train, initialization_method="estimated").fit()
    fcst7_test = holt1.forecast(len(y_test)).rename(r"Holt=%s" % round(es4.model.params["smoothing_level"],2))
    fcst7_train = holt1.forecast(len(y_train)).rename(r"Holt=%s" % round(es4.model.params["smoothing_level"],2))
    
    holt2 = Holt(y_train).fit(smoothing_level = 0.3, smoothing_trend = 0.1)
    fcst8_test = holt2.forecast(len(y_test)).rename("Holt_alpha=0.3_beta_0.1")
    fcst8_train = holt2.forecast(len(y_train)).rename("Holt_alpha=0.3_beta_0.1")
    
    #model_type = "arima"
    
    result = adfuller(y_train.values)
    p_value = result[1]
    
    if p_value > 0.05:  # If p-value is greater than 0.05, data is not stationary
        d = 1  # Differencing is needed
        diff_y = y_train.diff(periods=d).dropna()
    else:
        d = 0  # Data is already stationary
        diff_y = y_train
    
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
        print("P Value SD Fuller:" + str(round(p_value,3)))
        print("Best Order (p, d, q):", best_order)
        print("Best AIC:", round(best_aic,0))
    
    arima = ARIMA(y_train, order=best_order).fit()
    
    fcst9_test = arima.forecast(len(y_test)).rename("Arima_Best")
    fcst9_train = arima.forecast(len(y_train)).rename("Arima_Best")
    
    #seasonal naive 2
    fcst10_test = y_test.copy()
    fcst10_test[y_name] = y_train.iloc[-365][y_name]
    
    fcst10_train = y_train.copy()
    fcst10_train[y_name] = y_train.iloc[-365][y_name]
    
    fcst10_test = fcst10_test.rename(columns={y_name: "snaive"})
    fcst10_train = fcst10_train.rename(columns={y_name: "snaive"})
    
    #es damped
    es5 = Holt(y_train, initialization_method="estimated", damped_trend=True).fit()
    fcst11_test = es4.forecast(len(y_test)).rename(r"DampedES_alpha=%s" % round(es5.model.params["smoothing_level"],2))
    fcst11_train = es4.forecast(len(y_train)).rename(r"DampedES_alpha=%s" % round(es5.model.params["smoothing_level"],2))
    
    #Combined 

    comb_test = pd.concat([fcst8_test,  fcst3_test,  fcst11_test], axis = 1)
    comb_test["combined"] = comb_test.mean(axis = 1)

    comb_train = pd.concat([fcst8_train,  fcst3_train,  fcst11_train], axis = 1)
    comb_train["combined"] = comb_train.mean(axis = 1)
    
    fcst12_test = comb_test["combined"]
    fcst12_train = comb_train["combined"]

    for i in range(1, 13):

        if verbose == True:
            print(i)
            
        fcst_train = locals()['fcst' + str(i) + '_train'] 
        fcst_test = locals()['fcst' + str(i) + '_test']   
        
        try:
            model_name = fcst_test.name
        except AttributeError:
            model_name = fcst_test.columns[0]
        
        if verbose == True:
            print(model_name)
        
        train_rmse = math.sqrt(mean_squared_error(y_train, fcst_train))
        test_rmse = math.sqrt(mean_squared_error(y_test, fcst_test))
        
        # calculate smape
        train_smape = smape(y_train.values.flatten(), fcst_train.values.flatten())
        test_smape = smape(y_test.values.flatten(), fcst_test.values.flatten())
        
        # calculate mase
        train_mase = mean_absolute_percentage_error(y_train, fcst_train)
        test_mase = mean_absolute_percentage_error(y_test, fcst_test)
        
        
        if verbose == True:
            print(model_name)
            print('Train Score ' + str(i) + ': %.2f RMSE' % (train_rmse))
            print('Test Score ' + str(i) + ': %.2f RMSE' % (test_rmse))
            print('Train Score ' + str(i) + ': %.2f SMAPE' % (train_smape))
            print('Test Score ' + str(i) + ': %.2f SMAPE' % (test_smape))
            
            # Visualising the results
            figure, axes = plt.subplots()
            axes.xaxis_date()

            axes.plot(pd.to_datetime(y_test.index), y_test.values, color = 'red', label = 'Real')
            axes.plot(pd.to_datetime(y_test.index), fcst_test, color = 'blue', label = 'Predicted')
            plt.title(str(model_name) + xtra_desc + dte.strftime("%Y%m%d") + ' prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig("output/" + dte.strftime("%Y%m%d") + str(model_name) + xtra_desc  + '_prediction.png')
            plt.show()
        
        model_score = pd.DataFrame(data = {"model_name": model_name, "step": str(step), "dte": str(dte),
                                           "train_rmse": train_rmse, "test_rmse":test_rmse,
                                           "train_smape": train_smape, "test_smape":test_smape,
                                           "train_mase": train_mase, "test_mase":test_mase
                                           }, 
                                   index = [model_name + "_" + str(step) + "_" + str(dte)])
        
        
        fcst = pd.DataFrame(data = {"model_name": str(model_name), "step":[step] * len(y_test), 
                                    "dte" : y_test.index.to_list(), 
                                    "y_pred": fcst_test.values.flatten(), "y_obs": y_test.values.flatten()})
        
        fcst_y_only = pd.concat([fcst_y_only, fcst], axis = 0)
        model_score_y_only = pd.concat([model_score_y_only, model_score], axis = 0)
    

    return(fcst_y_only, model_score_y_only)
