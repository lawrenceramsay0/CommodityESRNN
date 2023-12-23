#!/usr/bin/env python
# coding: utf-8

# ## Set up

#TODO: Analyse residuals!
#TODO: Financial news check the headlines agaist a human samples
#TODO: Need to deseaonalise the data by day (7) and month (30/31)
#TODO: Apply Seasonality decomp and differenceing
# In[Setup]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import torch
from torch import nn
import skorch
import math
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from torch.nn.utils import prune
from datetime import date
from datetime import datetime, timedelta
import os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
from itertools import product


sys.path.append('ESRNN-GPU')

from es_rnn.model import DRNN

# from ESRNN import ESRNN
# from ESRNN.utils_evaluation import evaluate_prediction_owa
# from ESRNN.utils_visualization import plot_grid_prediction
# =============================================================================

root = os.getcwd()[-0:-6]
root

run_hyper_params_esrnn = False
run_hyper_params_net = False
train_neural_net = True
testing = True
forecast_start_date = date(2022, 12, 1)

y_remove = "tgt_crd"
y_name = "tgt_irn"
model_name = "lstm"
#target_cols = ["close_irn", "Close_crd", "high_irn", "low_irn", "high_crd", "low_crd"]

#data is pre-cleaned no no NA's should be present. This is to make sure.
datft = pd.read_csv("data/dat_ft.csv")
datft = datft.set_index("dte")
datft = datft.dropna() 

datr = pd.read_csv("data/dat_raw.csv")
datr = datr.set_index("dte")
datr = datr.dropna()

#dates = dat.iloc[:-200][(dat.iloc[:-200].index >= str(forecast_start_date))].index[::5]
dates = datr.iloc[(datr.index >= str(forecast_start_date))][:-57].index[::15].copy()
# dat = dat.drop("tgt_iron", axis=1)#.astype('float64')
datr.shape

#sys.path.append('C:/Users/lawre/OneDrive/Documents/GitHub/ESRNN-GPU')

sys.path.append('Python')
# =============================================================================
# from train_test_split_custom import feature_label_split, train_val_test_split, train_test_split_date
# from rnn import RNN
# from gru import GRU
# from lstm import LSTM
# from esrnn import ESRNN
# from subset import processSubset, forward, fwd_subset, vif_correlation_subset
# =============================================================================
from hyperopt import hyperopt
from train_neural_net import train_net
from train_y_only import train_y_only
from train_statistical import train_stat

fcst_all = pd.DataFrame(columns = ["model_name", "step", "dte", "y_pred", "y_obs"])
model_score_all = pd.DataFrame(columns = ["model_name", "step", "dte", "train_rmse", "test_rmse"])
model_score_all_hyper_esrnn = pd.DataFrame(columns = ["model_name", "step", "dte", "train_rmse", "test_rmse", "hyper", "hyperidx"])
model_score_all_hyper_net = pd.DataFrame(columns = ["model_name", "step", "dte", "train_rmse", "test_rmse", "hyper", "hyperidx"])


def expand_grid(dictionary):
    return(pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys()))


# In[Hyper ESRNN]:
hyper_epochs = 200
if run_hyper_params_esrnn == True:

    dte_str = dates[0]; 
    dte = datetime.strptime(dte_str, "%Y-%m-%d")
# =============================================================================
#     outer_params1 = {
#             'hidden_dim': [16],# 8], 32, 
#             'num_layers': [2], #,  4
#             'dropout_prob': [0.2], #, 0.4
#             'criterion': [torch.nn.MSELoss()], #torch.nn.L1Loss(), 
#             'prune':[0.2], #, 0.4
#             'alpha_smoothing':[0.2, 0.5, 0.7],
#             'cell_type':["GRU"], #"LSTM", 
#             'dilations':[((1, 7), (14, 28))],#, ((1, 3), (7, 14)), ((1, 3), (7, 14), (14, 28))], #((1, 4), (8, 12)), 
#             'multiplicative': [True, False],
#             'no_trend': [False],
#             'beta_smoothing':[0.2, 0.5, 0.7],
#             'seasonality': [7], #7,
#             'learning_rate': [0.001],
#             'subset': [True, False]
#         }
#     
#     outer_params2 = {
#             'hidden_dim': [16],# 8], 32, 
#             'num_layers': [2], #,  4
#             'dropout_prob': [0.2], #, 0.4
#             'criterion': [torch.nn.MSELoss()], #torch.nn.L1Loss(), 
#             'prune':[0.2], #, 0.4
#             'alpha_smoothing':[0.2, 0.5, 0.7],
#             'cell_type':["GRU"], #"LSTM", 
#             'dilations':[((1, 7), (14, 28))],#, ((1, 3), (7, 14)), ((1, 3), (7, 14), (14, 28))], #((1, 4), (8, 12)), 
#             'multiplicative': [True, False],
#             'no_trend': [True],
#             'beta_smoothing':[0.5],
#             'seasonality': [7], #7,
#             'learning_rate': [0.001],
#             'subset': [True, False]
#         }
# =============================================================================
    
    outer_params1 = {
            'hidden_dim': [16, 8, 32], 
            'num_layers': [2,  4],
            'dropout_prob': [0.2],
            'criterion': [torch.nn.MSELoss(), torch.nn.L1Loss()], 
            'prune':[0.2],
            'alpha_smoothing':[0.2],
            'cell_type':["GRU", "LSTM"], 
            'dilations':[((1, 3), (7, 14))], #((1, 4), (8, 12)), 
            'multiplicative': [True],
            'no_trend': [True],
            'beta_smoothing':[0.2],
            'seasonality': [7], #7,
            'learning_rate': [0.001, 0.005],
            'subset': [True]
        }
    
    output_pred_from_esrnn = False
    
    def create_output_string(model_name, hidden_dim, num_layers, dropout_prob, criterion, prune_prop,
                             alpha_smoothing, beta_smoothing, cell_type, dilations, multiplicative_seasonality,
                             no_trend, seasonality, subset
                             ):

        output_string = "Model Name:" + model_name + \
                        " Hidden Dim:" + str(hidden_dim) + \
                      " Num Layers:" + str(num_layers) + \
                      " Dropout:" + str(dropout_prob) + \
                      " Criterion:" + str(criterion) + \
                      " Prune:" + str(prune_prop) + \
                        " Alpha Smoothing:" + str(alpha_smoothing) + \
                        " Cell Type:" + str(cell_type) + \
                        " Dilations:" + str(dilations) + \
                        " Multiplicative Seasonality:" + str(multiplicative_seasonality) + \
                        " No Trend:" + str(no_trend) + \
                        " Beta Smoothing:" + str(beta_smoothing) + \
                        " Seasonality:" + str(seasonality) + \
                        " Subset:" + str(subset) 
        return(output_string)
            
#,expand_grid(outer_params2)
    
    outer_params_grid = pd.concat([expand_grid(outer_params1)], ignore_index=True)#.iloc[22:].reset_index()
    
    print(outer_params_grid)
    
    for idx, x in outer_params_grid.iterrows(): 
        
        outer_params = outer_params_grid
        
        if idx == 0 | ((outer_params_grid['subset'][idx] == False) & (idx <= 2)):
            read_subset_values = False
            read_pred_values = False
        else:
            read_subset_values = True
            read_pred_values = True
        
        fcst_esrnn_subset, model_score_esrnn_subset_hyper = train_net(dat = datr, dte = dte, step = 1,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, run_vif = False, 
                                                run_fwd_selection = outer_params_grid['subset'][idx], 
                                                read_subset_values = read_subset_values,
                                                num_epochs = hyper_epochs, 
                                                verbose = True,
                                                read_pred_values = read_pred_values,
                                                learning_rate = outer_params_grid['learning_rate'][idx],
                                                model_name = "esrnn",
                                                xtra_desc = "esrnnsubset"+str(outer_params_grid['subset'][idx])+"hyper",
                                                epoch_print = 20, 
                                                dropout = outer_params_grid['dropout_prob'][idx],
                                                esrnn_cell_type = outer_params_grid['cell_type'][idx],
                                                esrnn_alpha_smoothing = outer_params_grid['alpha_smoothing'][idx],
                                                esrnn_beta_smoothing = outer_params_grid['beta_smoothing'][idx],
                                                esrnn_dilations = outer_params_grid['dilations'][idx],
                                                prune_prop = outer_params_grid['prune'][idx],
                                                num_layers = outer_params_grid['num_layers'][idx],
                                                hidden_dim = outer_params_grid['hidden_dim'][idx],
                                                esrnn_no_trend = outer_params_grid['no_trend'][idx],
                                                esrnn_multiplicative_seasonality = outer_params_grid['multiplicative'][idx],
                                                output_pred_from_esrnn=output_pred_from_esrnn,
                                                rss_cutoff=5)
    
        output_string = create_output_string(model_name = model_name, hidden_dim = list(outer_params_grid.values[idx])[0],
            num_layers = list(outer_params_grid.values[idx])[1], dropout_prob = list(outer_params_grid.values[idx])[2],
            criterion = list(outer_params_grid.values[idx])[3], prune_prop = list(outer_params_grid.values[idx])[4],
            alpha_smoothing = list(outer_params_grid.values[idx])[5], cell_type = list(outer_params_grid.values[idx])[6],
            dilations = list(outer_params_grid.values[idx])[7], multiplicative_seasonality = list(outer_params_grid.values[idx])[8],
            no_trend = list(outer_params_grid.values[idx])[9], beta_smoothing = list(outer_params_grid.values[idx])[10],
            seasonality = list(outer_params_grid.values[idx])[11], subset = list(outer_params_grid.values[idx])[12])
        
        model_score_esrnn_subset_hyper["hyper"] = output_string
        model_score_esrnn_subset_hyper["hyperidx"] = idx
        model_score_esrnn_subset_hyper["epochs"] = hyper_epochs
        
        #Featured data hyperparams
        if outer_params_grid['subset'][idx] == True:
            fcst_esrnn_subset, model_score_esrnn_ft_hyper = train_net(dat = datft, dte = dte, step = 1,
                                                    y_name = y_name, y_remove = y_remove,
                                                    read_vif_values = False, run_vif = False, 
                                                    run_fwd_selection = True, 
                                                    read_subset_values = read_subset_values,
                                                    num_epochs = hyper_epochs, 
                                                    verbose = True,
                                                    read_pred_values = read_pred_values,
                                                    learning_rate = outer_params_grid['learning_rate'][idx],
                                                    model_name = "esrnn",
                                                    xtra_desc = "esrnnfthyper",
                                                    epoch_print = 20, 
                                                    dropout = outer_params_grid['dropout_prob'][idx],
                                                    esrnn_cell_type = outer_params_grid['cell_type'][idx],
                                                    esrnn_alpha_smoothing = outer_params_grid['alpha_smoothing'][idx],
                                                    esrnn_beta_smoothing = outer_params_grid['beta_smoothing'][idx],
                                                    esrnn_dilations = outer_params_grid['dilations'][idx],
                                                    prune_prop = outer_params_grid['prune'][idx],
                                                    num_layers = outer_params_grid['num_layers'][idx],
                                                    hidden_dim = outer_params_grid['hidden_dim'][idx],
                                                    esrnn_no_trend = outer_params_grid['no_trend'][idx],
                                                    esrnn_multiplicative_seasonality = outer_params_grid['multiplicative'][idx],
                                                    output_pred_from_esrnn=output_pred_from_esrnn,
                                                    max_subset_cols=30,
                                                    rss_cutoff=29)
    
            output_string = create_output_string(model_name = model_name, hidden_dim = list(outer_params_grid.values[idx])[0],
                num_layers = list(outer_params_grid.values[idx])[1], dropout_prob = list(outer_params_grid.values[idx])[2],
                criterion = list(outer_params_grid.values[idx])[3], prune_prop = list(outer_params_grid.values[idx])[4],
                alpha_smoothing = list(outer_params_grid.values[idx])[5], cell_type = list(outer_params_grid.values[idx])[6],
                dilations = list(outer_params_grid.values[idx])[7], multiplicative_seasonality = list(outer_params_grid.values[idx])[8],
                no_trend = list(outer_params_grid.values[idx])[9], beta_smoothing = list(outer_params_grid.values[idx])[10],
                seasonality = list(outer_params_grid.values[idx])[11], subset = list(outer_params_grid.values[idx])[13])
        
            model_score_esrnn_ft_hyper["hyper"] = output_string
            model_score_esrnn_ft_hyper["hyperidx"] = idx
            model_score_esrnn_ft_hyper["epochs"] = hyper_epochs
#model_score_esrnn_subset_hyper
        #fcst_all = pd.concat([fcst_esrnn_subset],axis = 0)
        model_score_all_hyper_esrnn = pd.concat([model_score_all_hyper_esrnn, model_score_esrnn_ft_hyper], axis = 0)
    
        model_score_all_hyper_esrnn.to_csv("output/" + date.today().strftime("%Y%m%d") + y_name + "pred" + str(output_pred_from_esrnn) + \
                                           "_model_score_hyper_esrnn.csv")
#fcst_all.to_csv("output/"  + date.today().strftime("%Y%m%d") + y_name +"_fcst_all_hyper.csv")

# In[Hyper Net]:

dataset_type = "FT"    

if dataset_type == "RAW":
    run_fwd_selection = False
    dat_net = datr
elif dataset_type == "SUBSET":
    run_fwd_selection = True
    dat_net = datr
elif dataset_type == "FT":
    run_fwd_selection = True
    dat_net = datft

if run_hyper_params_net == True:
    dte_str = dates[0]; 
    dte = datetime.strptime(dte_str, "%Y-%m-%d")
    outer_params_net = {
            'hidden_dim': [16, 8, 32], 
            'num_layers': [2,  4],
            'dropout_prob': [0.2, 0.4],
            'criterion': [torch.nn.MSELoss()], #torch.nn.L1Loss(), 
            'prune':[0.2, 0.4],
            'alpha_smoothing':[0.2, 0.5, 0.7],
            'learning_rate': [0.001, 0.005]
        }
    
    
    outer_params_grid_net = expand_grid(outer_params_net)#pd.concat([expand_grid(outer_params_net)], ignore_index=True)#.iloc[10:12].reset_index()
    
    print(outer_params_grid_net)
    
    for idx, x in outer_params_grid_net.iterrows(): 
        
        if idx == 0:
            read_subset_values = False
            read_pred_values = False
        else:
            read_subset_values = True
            read_pred_values = True
        
        fcst_lstm_subset, model_score_lstm_subset_hyper = train_net(dat = dat_net, dte = dte, step = 1,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, run_vif = False, 
                                                run_fwd_selection = run_fwd_selection, 
                                                read_subset_values = read_subset_values,
                                                num_epochs = 50, 
                                                verbose = True,
                                                read_pred_values = read_pred_values,
                                                learning_rate = outer_params_grid_net['learning_rate'][idx],
                                                model_name = "lstm",
                                                xtra_desc = "lstmhyper",
                                                epoch_print = 20, 
                                                dropout = outer_params_grid_net['dropout_prob'][idx],
                                                prune_prop = outer_params_grid_net['prune'][idx],
                                                num_layers = outer_params_grid_net['num_layers'][idx],
                                                hidden_dim = outer_params_grid_net['hidden_dim'][idx],
                                                output_pred_from_esrnn=False,
                                                max_subset_cols=30,
                                                rss_cutoff=29)
        
    
        output_string = "Model Name:" + model_name + \
                        " Hidden Dim:" + str(list(outer_params_grid_net.values[idx])[0]) + \
                      " Num Layers:" + str(list(outer_params_grid_net.values[idx])[1]) + \
                      " Dropout:" + str(list(outer_params_grid_net.values[idx])[2]) + \
                      " Criterion:" + str(list(outer_params_grid_net.values[idx])[3]) + \
                      " Prune:" + str(list(outer_params_grid_net.values[idx])[4])
        
        model_score_lstm_subset_hyper["hyper"] = output_string
        model_score_lstm_subset_hyper["hyperidx"] = idx
        
        fcst_gru_subset, model_score_gru_subset_hyper = train_net(dat = dat_net, dte = dte, step = 1,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, run_vif = False, 
                                                run_fwd_selection = run_fwd_selection, 
                                                read_subset_values = read_subset_values,
                                                num_epochs = 50, 
                                                verbose = True,
                                                read_pred_values = read_pred_values,
                                                learning_rate = outer_params_grid_net['learning_rate'][idx],
                                                model_name = "gru",
                                                xtra_desc = "gruhyper",
                                                epoch_print = 20, 
                                                dropout = outer_params_grid_net['dropout_prob'][idx],
                                                prune_prop = outer_params_grid_net['prune'][idx],
                                                num_layers = 2,
                                                hidden_dim = outer_params_grid_net['hidden_dim'][idx],
                                                output_pred_from_esrnn=False,
                                                max_subset_cols=30,
                                                rss_cutoff=29)
        
    
        output_string = "Model Name:" + model_name + \
                        " Hidden Dim:" + str(list(outer_params_grid_net.values[idx])[0]) + \
                      " Num Layers:" + str(list(outer_params_grid_net.values[idx])[1]) + \
                      " Dropout:" + str(list(outer_params_grid_net.values[idx])[2]) + \
                      " Criterion:" + str(list(outer_params_grid_net.values[idx])[3]) + \
                      " Prune:" + str(list(outer_params_grid_net.values[idx])[4])
        
        model_score_gru_subset_hyper["hyper"] = output_string
        model_score_gru_subset_hyper["hyperidx"] = idx

        fcst_rnn_subset, model_score_rnn_subset_hyper = train_net(dat = dat_net, dte = dte, step = 1,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, run_vif = False, 
                                                run_fwd_selection = run_fwd_selection, 
                                                read_subset_values = read_subset_values,
                                                num_epochs = 50, 
                                                verbose = True,
                                                read_pred_values = read_pred_values,
                                                learning_rate = outer_params_grid_net['learning_rate'][idx],
                                                model_name = "rnn",
                                                xtra_desc = "rnnhyper",
                                                epoch_print = 20, 
                                                dropout = outer_params_grid_net['dropout_prob'][idx],
                                                prune_prop = outer_params_grid_net['prune'][idx],
                                                num_layers = outer_params_grid_net['num_layers'][idx],
                                                hidden_dim = outer_params_grid_net['hidden_dim'][idx],
                                                output_pred_from_esrnn=False,
                                                max_subset_cols=30,
                                                rss_cutoff=29)
        
    
        output_string = "Model Name:" + model_name + \
                        " Hidden Dim:" + str(list(outer_params_grid_net.values[idx])[0]) + \
                      " Num Layers:" + str(list(outer_params_grid_net.values[idx])[1]) + \
                      " Dropout:" + str(list(outer_params_grid_net.values[idx])[2]) + \
                      " Criterion:" + str(list(outer_params_grid_net.values[idx])[3]) + \
                      " Prune:" + str(list(outer_params_grid_net.values[idx])[4])
        
        model_score_rnn_subset_hyper["hyper"] = output_string
        model_score_rnn_subset_hyper["hyperidx"] = idx

        #fcst_all = pd.concat([fcst_esrnn_subset],axis = 0)
        model_score_all_hyper_net = pd.concat([model_score_all_hyper_net, model_score_lstm_subset_hyper, model_score_gru_subset_hyper,
                                               model_score_rnn_subset_hyper], axis = 0)
    
        model_score_all_hyper_net.to_csv("output/" + date.today().strftime("%Y%m%d") + y_name + dataset_type + "_model_score_hyper_net.csv")
#fcst_all.to_csv("output/"  + date.today().strftime("%Y%m%d") + y_name +"_fcst_all_hyper.csv")


# In[Train Net]:
epochs = 200
epoch_print = 10
esrnn_epochs = 600

if train_neural_net == True:

    for i, dte_str in enumerate(dates):
        #dte_str = dates[0]; i = 1
        print(dte_str)
        dte = datetime.strptime(dte_str, "%Y-%m-%d")
        fcst_y_only, model_score_y_only = train_y_only(dat = datr, 
                                                 dte = dte, 
                                                 step = i,
                                                 y_name = y_name,
                                                 verbose = True)
        
        
        fcst_stat, model_score_stat = train_stat(dat = datft, 
                                                 dte = dte, 
                                                 step = i,
                                                 y_name = y_name,
                                                 y_remove = y_remove,
                                                 run_vif = False,
                                                 run_fwd_selection = True,
                                                 read_vif_values = False,
                                                 read_subset_values = False,
                                                 read_pred_values=False,
                                                 verbose = True,
                                                 model_name = "ols",
                                                 xtra_desc="stat",
                                                 max_subset_cols=30,
                                                 rss_cutoff=29)
        
        fcst_arima_sea, model_arima_sea = train_stat(dat = datft, 
                                                 dte = dte, 
                                                 step = i,
                                                 y_name = y_name,
                                                 y_remove = y_remove,
                                                 run_vif = False,
                                                 run_fwd_selection = True,
                                                 read_vif_values = False,
                                                 read_subset_values = False,
                                                 read_pred_values=False,
                                                 verbose = True,
                                                 model_name = "arima_sea",
                                                 xtra_desc="arima_sea",
                                                 max_subset_cols=30,
                                                 rss_cutoff=29)
        
        fcst_arima_xreg, model_arima_xreg = train_stat(dat = datft, 
                                                 dte = dte, 
                                                 step = i,
                                                 y_name = y_name,
                                                 y_remove = y_remove,
                                                 run_vif = False,
                                                 run_fwd_selection = True,
                                                 read_vif_values = False,
                                                 read_subset_values = False,
                                                 read_pred_values=False,
                                                 verbose = True,
                                                 model_name = "arima_xreg",
                                                 xtra_desc="arima_xreg",
                                                 max_subset_cols=30,
                                                 rss_cutoff=29)
        
        fcst_gru, model_score_gru = train_net(dat = datr, dte = dte, step = i,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, read_subset_values = False,
                                                run_vif = False, run_fwd_selection=True,
                                                num_epochs = 150, verbose = True,
                                                read_pred_values=False,
                                                learning_rate = 0.001,
                                                num_layers = 2,
                                                epoch_print = epoch_print,
                                                model_name = "gru",
                                                xtra_desc = "grusubsettest",
                                                hidden_dim=16,
                                                prune_prop=0.2,
                                                dropout=0.4,
                                                rss_cutoff=29)
        
        fcst_lstm, model_score_lstm = train_net(dat = datr, dte = dte, step = i,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, read_subset_values = False,
                                                run_vif = False, run_fwd_selection=True,
                                                num_epochs = 175, verbose = True,
                                                read_pred_values=False,
                                                learning_rate = 0.001,
                                                num_layers = 2,
                                                epoch_print = epoch_print,
                                                model_name = "lstm",
                                                xtra_desc = "lstmsubset",
                                                hidden_dim=16,
                                                prune_prop=0.2,
                                                dropout=0.4,
                                                rss_cutoff=29)
        
        fcst_rnn, model_score_rnn = train_net(dat = datr, dte = dte, step = i,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, read_subset_values = False,
                                                run_vif = False, run_fwd_selection=True,
                                                num_epochs = 200, verbose = True,
                                                read_pred_values=False,
                                                learning_rate = 0.001,
                                                num_layers = 2,
                                                epoch_print = epoch_print,
                                                model_name = "rnn",
                                                xtra_desc = "rnnsubset",
                                                hidden_dim=16,
                                                prune_prop=0.2,
                                                dropout=0.4,
                                                rss_cutoff=29)
        
        fcst_esrnn_subset, model_score_esrnn_subset = train_net(dat = datft, dte = dte, step = i,
                                                y_name = y_name, y_remove = y_remove,
                                                read_vif_values = False, read_subset_values = True,
                                                run_vif = False, run_fwd_selection = True,
                                                num_epochs = 200, 
                                                verbose = True,
                                                read_pred_values=True,
                                                learning_rate = 0.001,
                                                model_name = "esrnn",
                                                xtra_desc = "ftsubsettest",
                                                epoch_print = epoch_print, 
                                                esrnn_cell_type = "RNN",
                                                esrnn_alpha_smoothing = 0.2,
                                                esrnn_dilations = ((1, 3), (7, 14)),
                                                output_pred_from_esrnn=False,
                                                esrnn_no_trend=True,
                                                esrnn_beta_smoothing=0.2,
                                                dropout=0.2,
                                                hidden_dim=16,
                                                num_layers=2,
                                                prune_prop=0.2,
                                                max_subset_cols=30,
                                                rss_cutoff=29, 
                                                seasonality = 7)
    
        fcst_all = pd.concat([fcst_all, 
                              fcst_stat,
                              fcst_arima_sea,
                              fcst_arima_xreg,
                              fcst_y_only,
                              fcst_esrnn_subset,
                              fcst_gru,
                              fcst_rnn,
                              fcst_lstm
                              ],axis = 0)
        
        model_score_all = pd.concat([model_score_all, 
                                     model_arima_sea,
                                     model_arima_xreg,
                                     model_score_y_only,
                                     model_score_esrnn_subset,
                                     model_score_gru, 
                                     model_score_lstm,
                                     model_score_rnn,
                                     model_score_stat
                                     ], axis = 0)
        
        model_score_all.to_csv("output/" + date.today().strftime("%Y%m%d") + y_name + "_model_score.csv")
        fcst_all.to_csv("output/"  + date.today().strftime("%Y%m%d") + y_name +"_fcst_all.csv")

# In[Notes]

# =============================================================================
# The seasonality in the crude examples is a lot less prominent
#   This means that the data is more random and the esrnn wasnt designed for this 
#   Type of analysis
# =============================================================================




# In[Testing]

if testing == True:
    epochs = 100
    dropout = 0.0
    epoch_print = 10
    esrnn_epochs = 10
    dte_str = dates[0]; i = 1
    dte = datetime.strptime(dte_str, "%Y-%m-%d")
    
# =============================================================================
#     fcst_stat, model_score_stat = train_stat(dat = datft, 
#                                              dte = dte, 
#                                              step = i,
#                                              y_name = y_name,
#                                              y_remove = y_remove,
#                                              xtra_desc = "stat",
#                                              run_vif = False,
#                                              run_fwd_selection = True,
#                                              read_vif_values = False,
#                                              read_subset_values = True,
#                                              verbose = True,
#                                              run_corr_cutoff = True,
#                                              read_corr_values = False,
#                                              read_pred_values = True)
# =============================================================================
    
    fcst_esrnn_subset, model_score_esrnn_subset = train_net(dat = datr, dte = dte, step = i,
                                            y_name = y_name, y_remove = y_remove,
                                            read_vif_values = False, run_vif = False, 
                                            run_fwd_selection = True, read_subset_values = True,
                                            num_epochs = esrnn_epochs, verbose = True,
                                            read_pred_values = True,
                                            learning_rate = 0.001,
                                            model_name = "esrnn",
                                            xtra_desc = "esrnnsubset",
                                            epoch_print = epoch_print, 
                                            dropout = dropout,
                                            esrnn_cell_type = "GRU",
                                            esrnn_alpha_smoothing = 0.5,
                                            esrnn_dilations = ((1, 7), (14, 28)),
                                            esrnn_no_trend = True,
                                            esrnn_multiplicative_seasonality = True,
                                            output_pred_from_esrnn=False)
    
    #Know this works
# =============================================================================
#     fcst_lstm_raw, model_score_lstm_raw = train_net(dat = datr, dte = dte, step = i,
#                                             y_name = y_name, y_remove = y_remove,
#                                             read_vif_values = False, read_subset_values = False,
#                                             run_vif = False, run_fwd_selection=False,
#                                             num_epochs = epochs, verbose = True,
#                                             learning_rate = 0.001,
#                                             model_name = "lstm",
#                                             xtra_desc = "raw",
#                                             epoch_print = epoch_print, 
#                                             dropout = 0.2,
#                                             hidden_dim = 32,
#                                             num_layers = 4)
#     
#     fcst_lstm_all, model_score_lstm_all = train_net(dat = datft, dte = dte, step = i,
#                                         y_name = y_name, y_remove = y_remove,
#                                         read_vif_values = False, read_subset_values = False,
#                                         run_vif = False, run_fwd_selection = False,
#                                         read_pred_values = False,
#                                         num_epochs = epochs, verbose = True,
#                                         learning_rate = 0.001,
#                                         model_name = "lstm",
#                                         xtra_desc = "all",
#                                         epoch_print = epoch_print, 
#                                         dropout = 0.2,
#                                         hidden_dim = 32,
#                                         num_layers = 4)
#     
#     fcst_lstm_fwd, model_score_lstm_fwd = train_net(dat = datr, dte = dte, step = i,
#                                         y_name = y_name, y_remove = y_remove,
#                                         read_vif_values = False, read_subset_values = False,
#                                         run_vif = False, run_fwd_selection = True,
#                                         read_pred_values = False,
#                                         run_corr_cutoff = False,
#                                         read_corr_values = False,
#                                         num_epochs = epochs, verbose = True,
#                                         learning_rate = 0.001,
#                                         model_name = "lstm",
#                                         xtra_desc = "fwd",
#                                         epoch_print = epoch_print, 
#                                         dropout = 0.2,
#                                         hidden_dim = 32,
#                                         num_layers = 4)
#     
#     fcst_lstm_corr, model_score_lstm_corr = train_net(dat = datr, dte = dte, step = i,
#                                         y_name = y_name, y_remove = y_remove,
#                                         read_vif_values = False, read_subset_values = False,
#                                         run_vif = False, run_fwd_selection = False,
#                                         read_pred_values = False,
#                                         run_corr_cutoff = True,
#                                         read_corr_values = False,
#                                         num_epochs = epochs, verbose = True,
#                                         learning_rate = 0.001,
#                                         model_name = "lstm",
#                                         xtra_desc = "corr",
#                                         epoch_print = epoch_print, 
#                                         dropout = 0.2,
#                                         hidden_dim = 32,
#                                         num_layers = 4)
# =============================================================================
