# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:56:30 2023

@author: lawre
"""

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from skorch import NeuralNetRegressor
from train_test_split_custom import feature_label_split, train_val_test_split, train_val_test_split_date
from subset import processSubset, forward, fwd_subset, vif_correlation_subset
from torch import nn
import torch
import time
from skopt import BayesSearchCV
from rnn import RNN
from gru import GRU
from lstm import LSTM
from esrnn import ESRNN
from prune import prune_model_l1_unstructured
from numba import jit, cuda
from datetime import date
import numpy as np
from prep_data import prep_data
#from esrnn import ESRNN


def hyperopt(dat, y_name, 
             y_remove,
             outer_params_grid, 
             inner_params,
             dte,
             test_days = 50,
             val_days = 50,
             seasonality = 7,
             xtra_desc = "hyper",
             model_name = "lstm", 
             read_vif_values = False,
            verbose = True,
            read_subset_values = False,
            run_vif = True,
            run_fwd_selection = True,
            run_corr_cutoff = False,
            read_pred_values = False,
            hyp_type = ""
            ):
    
    dat = dat.copy().drop(y_remove, axis=1)#.astype('float64')
    
    test_days = test_days + seasonality
    val_days = val_days + seasonality
   
    for idx, x in enumerate(outer_params_grid): 
        
        outer_params = outer_params_grid
        
        #print(idx + outer_params[idx])
   
        test_days_loop = test_days + outer_params['seasonality'][idx] 

        if run_fwd_selection:
            X_train, X_val, X_test, y_train, y_val, y_test, reg_model, subsets = prep_data(dat = dat, y_name = y_name, dte = dte, 
                                                                                           read_pred_values = read_pred_values,
                                                                       val_days = val_days, test_days = test_days_loop, xtra_desc = xtra_desc,
                                                                       run_fwd_selection = run_fwd_selection, read_vif_values = read_vif_values,
                                                                       read_subset_values = read_subset_values) 
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = prep_data(dat = dat, y_name = y_name, dte = dte, read_pred_values = read_pred_values,
                                                                       val_days = val_days, test_days = test_days_loop, xtra_desc = xtra_desc,
                                                                       run_fwd_selection = run_fwd_selection, read_vif_values = read_vif_values,
                                                                       read_subset_values = read_subset_values) 
       
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_arr = scaler.fit_transform(X_train)
        X_val_arr = scaler.transform(X_val)
        X_test_arr = scaler.transform(X_test)
    
        y_train_arr = scaler.fit_transform(y_train)
        y_val_arr = scaler.transform(y_val)
        y_test_arr = scaler.transform(y_test)
        
        # make training and test sets in torch
        X_train_subset_tensor = torch.from_numpy(X_train_arr).type(torch.Tensor)
        X_test_subset_tensor = torch.from_numpy(X_test_arr).type(torch.Tensor)
        X_val_subset_tensor = torch.from_numpy(X_val_arr).type(torch.Tensor)
        y_train_tensor = torch.from_numpy(y_train_arr).type(torch.Tensor)
        y_test_tensor = torch.from_numpy(y_test_arr).type(torch.Tensor)
        y_val_tensor = torch.from_numpy(y_val_arr).type(torch.Tensor)
        
        X_train_subset_tensor = X_train_subset_tensor[:,:,None]
        X_test_subset_tensor = X_test_subset_tensor[:,:,None]
        X_val_subset_tensor = X_val_subset_tensor[:,:,None]
        
        start_time = time.time()   
        output = []
        

# =============================================================================
#         print("Hidden Dim:" + str(outer_params['hidden_dim'][idx]) + 
#               " Layers:" + str(outer_params['num_layers'][idx]) + 
#              " Dropout:" + str(outer_params['dropout_prob'][idx]) + 
#              " Criterion:" + str(outer_params['criterion'][idx]) +           
#              " Smoothing:" + str(outer_params['smoothing'][idx]) + 
#             " Cell Type:" + str(outer_params['cell_type'][idx]) + 
#             " Dilations:" + str(outer_params['dilations'][idx]))
# =============================================================================

        if model_name == "lstm":
            net = LSTM(input_dim = 1,
                        hidden_dim = outer_params['hidden_dim'][idx],
                        num_layers = outer_params['num_layers'][idx], 
                        output_dim = 1,
                        dropout_prob = outer_params['dropout_prob'][idx])
        
        elif model_name == "gru":           
            net = GRU(input_dim = 1,
                        hidden_dim = outer_params['hidden_dim'][idx],
                        num_layers = outer_params['num_layers'][idx], 
                        output_dim = 1,
                        dropout_prob = outer_params['dropout_prob'][idx])
            
        elif model_name == "rnn":
            net = RNN(input_dim = 1,
                        hidden_dim = outer_params['hidden_dim'][idx],
                        num_layers = outer_params['num_layers'][idx], 
                        output_dim = 1,
                        dropout_prob = outer_params['dropout_prob'][idx])
        elif model_name == "esrnn":
            net = ESRNN(input_dim=1, hidden_dim=outer_params['hidden_dim'][idx], output_dim=1, 
                              num_layers=outer_params['num_layers'][idx], 
                              dropout_prob = outer_params['dropout_prob'][idx], 
                              num_series = X_train_subset_tensor.shape[1],
                              output_window_size = X_test_subset_tensor.shape[0] - outer_params['seasonality'][idx], 
                              state_hsize = X_test_subset_tensor.shape[0] - outer_params['seasonality'][idx],
                              add_nl_layer=True,
                              dilations = outer_params['dilations'][idx],
                              alpha_smoothing = outer_params['alpha_smoothing'][idx],
                              beta_smoothing = outer_params['beta_smoothing'][idx],
                              rnn_cell_type=outer_params['cell_type'][idx],
                              no_trend=outer_params['no_trend'][idx],
                              multiplicative_seasonality=outer_params['multiplicative'][idx],
                              output_prediction=False,
                              seasonality=outer_params['seasonality'][idx]
                              )

        net = prune_model_l1_unstructured(net, nn.Conv2d, outer_params['prune'][idx])

        net = NeuralNetRegressor(net,
                                 verbose=1, 
                                criterion = outer_params['criterion'][idx],
                                optimizer= torch.optim.Adam)

        #criterion = nn.L1Loss()
        #criteion = nn.MSELoss()
        #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
        #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

#         X_trf = np.array(X_train)
#         y_trf = np.array(y_train).reshape(-1, 1)

        class TimeSeriesSplitByTime:
            def __init__(self, n_splits):
                self.n_splits = n_splits
        
            def split(self, X, y=None, groups=None):
                n = len(X)
                k = int(n / self.n_splits)
                for i in range(self.n_splits):
                    start = i * k
                    end = (i + 1) * k
                    train_indices = list(range(start, end))
                    test_indices = list(range(end, min(end + k, n)))
                    yield train_indices, test_indices  # Return two values
        
            def get_n_splits(self, X, y, groups=None):
                return self.n_splits
        
        # Usage:
        tscv = TimeSeriesSplitByTime(n_splits=2)
        
        gs = hyper_train(net = net, inner_params = inner_params, 
                         X_train_subset_tensor = X_train_subset_tensor, 
                         y_train_tensor = y_train_tensor, tscv = tscv)

        if model_name == "esrnn":

            output_string = "Model Name:" + model_name + \
                            " Hidden Dim:" + str(list(outer_params.values[idx])[0]) + \
                          " Num Layers:" + str(list(outer_params.values[idx])[1]) + \
                          " Dropout:" + str(list(outer_params.values[idx])[2]) + \
                          " Criterion:" + str(list(outer_params.values[idx])[3]) + \
                          " Prune:" + str(list(outer_params.values[idx])[4]) + \
                            " Alpha Smoothing:" + str(list(outer_params.values[idx])[5]) + \
                            " Cell Type:" + str(list(outer_params.values[idx])[6]) + \
                            " Dilations:" + str(list(outer_params.values[idx])[7]) + \
                            " Multiplicative Seasonality:" + str(list(outer_params.values[idx])[8]) + \
                            " No Trend:" + str(list(outer_params.values[idx])[9]) + \
                            " Beta Smoothing:" + str(list(outer_params.values[idx])[10]) + \
                            " Seasonality:" + str(list(outer_params.values[idx])[11]) + \
                          str(gs.best_params_) + \
                          " Error: " + str(round(gs.cv_results_['mean_test_score'][gs.best_index_],2))
                      
        else:
            
            output_string = "Model Name:" + model_name + \
                            " Hidden Dim:" + str(list(outer_params.values[idx])[0]) + \
                          " Num Layers:" + str(list(outer_params.values[idx])[1]) + \
                          " Dropout:" + str(list(outer_params.values[idx])[2]) + \
                          " Criterion:" + str(list(outer_params.values[idx])[3]) + \
                          " Prune:" + str(list(outer_params.values[idx])[4]) + \
                          str(gs.best_params_) + \
                          " Error: " + str(round(gs.cv_results_['mean_test_score'][gs.best_index_],2))

        output.append(output_string)
        print(output_string)


    print(output)
    np.savetxt("output/" + date.today().strftime("%Y%m%d") + y_name + "_" + hyp_type + "_hyperopt.csv",
            output,
            delimiter =", ",
            fmt ='% s')

    #output.to_csv("output/" + date.today().strftime("%Y%m%d") + y_name + "_" + hyp_type + "_hyperopt.csv")
    print("--- %s seconds ---" % (time.time() - start_time))
    print(str(gs.best_params_) + " with " + str(round(gs.cv_results_['mean_test_score'][gs.best_index_],2)))
    
jit(target_backend='cuda')  
def hyper_train(net, inner_params, X_train_subset_tensor, y_train_tensor, tscv):
    
    gs = GridSearchCV(net, inner_params, refit=False, scoring='neg_root_mean_squared_error', 
                      verbose=1, error_score='raise', cv=2, n_jobs = -1) #TimeSeriesSplit(n_splits=2)

    gs.fit(X_train_subset_tensor, y_train_tensor)
    
    #print(gs)
    
    return(gs)
