# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:45:42 2023

@author: lawre
"""

from train_test_split_custom import feature_label_split, train_val_test_split, train_val_test_split_date, train_val_test_split_idx
from subset import processSubset, forward, fwd_subset, vif_correlation_subset, corr_cutoff
import matplotlib.pyplot as plt
import torch
from rnn import RNN
from gru import GRU
from lstm import LSTM
from esrnn import ESRNN
import numpy as np
import pandas as pd
import time
from prune import prune_model_l1_unstructured
from torch import nn
from numba import jit, cuda
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from predict_features import predict_features
from error_metrics import smape
from prep_data import prep_data

def train_net(
            dat,
            dte,
            y_name,
            y_remove,
            step,
            model_name = "lstm",
            xtra_desc = "",
            input_dim = 1,
            hidden_dim = 32,
            num_layers = 4,
            output_dim = 1,
            learning_rate = 0.001,
            weight_decay = 0.000001,
            dropout = 0.2,
            prune_prop = 0.2,
            num_epochs = 200,
            read_vif_values = False,
            verbose = True,
            read_subset_values = False,
            run_vif = True,
            run_fwd_selection = True,
            val_days = 50,
            test_days = 50,
            seasonality = 7,
            epoch_print = 5,
            esrnn_cell_type = "LSTM",
            esrnn_alpha_smoothing = 0.5,
            esrnn_beta_smoothing = 0.2,
            esrnn_dilations = ((1, 7), (14, 28)),
            read_pred_values = False,
            run_corr_cutoff = True,
            read_corr_values = False,
            output_pred_from_esrnn = True,
            esrnn_no_trend = True,
            esrnn_multiplicative_seasonality = True,
            max_subset_cols = None,
            rss_cutoff = 5):
    
    #Split data
    #Train test validation splits
    
    test_days = test_days + seasonality
    val_days = val_days + seasonality

    dat = dat.copy().drop(y_remove, axis=1)#.astype('float64')
    
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
       
    
# =============================================================================
#     X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_idx(df = dat, target_col = y_name, dte = dte,
#                                                                               val_days = val_days, test_days = test_days)
# 
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
# =============================================================================
            
    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')
    
        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        
    early_stopper = EarlyStopper(patience=3, min_delta=10)

    scaler = MinMaxScaler()
    #Create tensors from subset of data
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test_pred)

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

    if verbose == True:
        print('x_train.shape = ',X_train_subset_tensor.shape)
        print('y_train.shape = ',y_train_tensor.shape)
        print('x_test.shape = ',X_test_subset_tensor.shape)
        print('y_test.shape = ',y_test_tensor.shape)
        print('x_val.shape = ',X_val_subset_tensor.shape)
        print('y_val.shape = ',y_val_tensor.shape)

    if model_name == "lstm":
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                  num_layers=num_layers, dropout_prob = dropout)
    elif model_name == "gru":
        model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                    num_layers=num_layers)
    elif model_name == "rnn":
        model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                          layer_dim=num_layers, dropout_prob = dropout)
    elif model_name == "esrnn":
        model = ESRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                          num_layers=num_layers, dropout_prob = dropout, num_series = X_train_subset_tensor.shape[1],
                          output_window_size = X_test_subset_tensor.shape[0] - seasonality, seasonality = seasonality,
                          state_hsize = X_test_subset_tensor.shape[0] - seasonality,
                          add_nl_layer=True,
                          dilations = esrnn_dilations,
                          alpha_smoothing = esrnn_alpha_smoothing,
                          beta_smoothing = esrnn_beta_smoothing,
                          rnn_cell_type = esrnn_cell_type,
                          no_trend = esrnn_no_trend,
                          multiplicative_seasonality = esrnn_multiplicative_seasonality,
                          output_prediction = output_pred_from_esrnn
                          )

    model = prune_model_l1_unstructured(model, nn.Conv2d, prune_prop)

    loss_fn = torch.nn.L1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if verbose == True:
        #print(model)
        print(model_name)
        #print(len(list(model.parameters())))
# =============================================================================
#         for i in range(len(list(model.parameters()))):
#             print(list(model.parameters())[i].size())
# =============================================================================
        
    # Train model
    #####################
    hist = np.zeros(num_epochs)
    hist_test = np.zeros(num_epochs)
    look_back = 10 # choose sequence length

    # Number of steps to unroll
    #seq_dim = look_back-1  

    start_time = time.time() 
    
    optimiser, hist, hist_test, y_train_pred = train_epochs(
        num_epochs = num_epochs, 
        X_train_subset_tensor = X_train_subset_tensor, y_train_tensor = y_train_tensor, 
        X_test_subset_tensor = X_test_subset_tensor, y_test_tensor = y_test_tensor,
        model = model, loss_fn = loss_fn, verbose = verbose, 
        epoch_print = epoch_print, hist = hist, hist_test = hist_test, 
        early_stopper = early_stopper, optimiser = optimiser,
        model_name = model_name, output_pred_from_esrnn = output_pred_from_esrnn)
    
# =============================================================================
#     for t in range(num_epochs):
#         # Initialise hidden state
#         # Don't do this if you want your LSTM to be stateful
#         #model.hidden = model.init_hidden()
#         #print(t)
#         # Forward pass
#         y_train_pred = model(X_train_subset_tensor)
#         y_test_pred = model(X_test_subset_tensor)
# 
#         loss = loss_fn(y_train_pred, y_train_tensor)
#         loss_test = loss_fn(y_test_pred, y_test_tensor)
#         if verbose == True:
#             if t % epoch_print == 0:# and t !=0:
#                 print("Epoch ", t, "MSE: ", loss.item())
#                 print("Epoch ", t, "Test MSE: ", loss_test.item())
#         hist[t] = loss.item()
#         hist_test[t] = loss_test.item()
#         
#         if early_stopper.early_stop(loss_test):          
#             print("Hit Early Stop")
#             break
# 
#         # Zero out gradient, else they will accumulate between epochs
#         optimiser.zero_grad()
# 
#         # Backward pass
#         loss.backward()
# 
#         # Update parameters
#         optimiser.step()
# =============================================================================
    
    if verbose == True:
        print("--- %s seconds ---" % (time.time() - start_time))
     
    if verbose == True:
        plt.plot(hist, label="Training loss")
        plt.plot(hist_test, label="Test loss")
        plt.legend()
        plt.title(str(model_name) + " " + xtra_desc + dte.strftime("%Y%m%d") + ' loss')
        plt.savefig("output/" + dte.strftime("%Y%m%d") + str(model_name) + xtra_desc  + '_loss.png')
        plt.show()

    # make test predictions
    if model_name == "esrnn" and output_pred_from_esrnn == True:
        y_test_pred = model(X_test_subset_tensor)[0]
        y_val_pred = model(X_val_subset_tensor)[0]
    else:
        y_test_pred = model(X_test_subset_tensor)
        y_val_pred = model(X_val_subset_tensor)
        
    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train_det = scaler.inverse_transform(y_train_tensor.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test_det = scaler.inverse_transform(y_test_tensor.detach().numpy())

    # invert predictions
    y_val_pred = scaler.inverse_transform(y_val_pred.detach().numpy())
    y_val_det = scaler.inverse_transform(y_val_tensor.detach().numpy())

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(y_train_det[:,0], y_train_pred[:,0]))
    test_rmse = math.sqrt(mean_squared_error(y_test_det[:,0], y_test_pred[:,0]))
    val_rmse = math.sqrt(mean_squared_error(y_val_det[:,0], y_val_pred[:,0]))
    
    # calculate smape
    train_smape = smape(y_train_det[:,0], y_train_pred[:,0])
    test_smape = smape(y_test_det[:,0], y_test_pred[:,0])
    val_smape = smape(y_val_det[:,0], y_val_pred[:,0])
    
    # calculate mase
    train_mase = mean_absolute_percentage_error(y_train_det[:,0], y_train_pred[:,0])
    test_mase = mean_absolute_percentage_error(y_test_det[:,0], y_test_pred[:,0])
    val_mase = mean_absolute_percentage_error(y_val_det[:,0], y_val_pred[:,0])
    
    if verbose == True:
        print('Test Score: %.2f RMSE' % (test_rmse))
        print('Train Score: %.2f RMSE' % (train_rmse))
        print('Validation Score: %.2f RMSE' % (val_rmse))
        
        # Visualising the results
        figure, axes = plt.subplots()
        axes.xaxis_date()

        axes.plot(pd.to_datetime(y_test.index), y_test.values, color = 'red', label = 'Real')
        axes.plot(pd.to_datetime(y_test.index), y_test_pred, color = 'blue', label = 'Predicted')
        plt.title(str(model_name) + " " + xtra_desc + dte.strftime("%Y%m%d") + ' prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig("output/" + dte.strftime("%Y%m%d") + str(model_name) + xtra_desc  + '_prediction.png')
        plt.show()


    fcst_y_only = pd.DataFrame()
    model_score_y_only = pd.DataFrame()
    
    model_score = pd.DataFrame(data = {"model_name": str(model_name) + "_" + str(xtra_desc), "step": str(step), "dte": str(dte),
                                       "train_rmse": train_rmse, "test_rmse":test_rmse,
                                       "train_smape": train_smape, "test_smape":test_smape,
                                       "train_mase": train_mase, "test_mase":test_mase
                                       }, 
                               index = [model_name + "_" + str(step) + "_" + str(dte)])
    
    fcst = pd.DataFrame(data = {"model_name": str(model_name) + "_" + str(xtra_desc), "step":[step] * len(y_test), 
                                "dte" : y_test.index.to_list(), 
                                "y_pred": y_test_pred.flatten(), "y_obs": y_test.values.flatten()})
    
    fcst_y_only = pd.concat([fcst_y_only, fcst], axis = 0)
    model_score_y_only = pd.concat([model_score_y_only, model_score], axis = 0)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return(fcst_y_only, model_score_y_only)

jit(target_backend='cuda')   
def train_epochs(num_epochs, X_train_subset_tensor, y_train_tensor, X_test_subset_tensor, y_test_tensor,
                 model, loss_fn, verbose, epoch_print, hist, hist_test, early_stopper, optimiser, model_name,
                 output_pred_from_esrnn = True):
     
    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()
        #print(t)
        # Forward pass
        if model_name == "esrnn" and output_pred_from_esrnn == True:
            y_train_pred, X_test_pred = model(X_train_subset_tensor)
            y_test_pred = model(X_test_pred)[0]
        else:
            y_train_pred = model(X_train_subset_tensor)
            y_test_pred = model(X_test_subset_tensor)

        loss = loss_fn(y_train_pred, y_train_tensor)
        loss_test = loss_fn(y_test_pred, y_test_tensor)
        if verbose == True:
            if t % epoch_print == 0:# and t !=0:
                print("Epoch ", t, "MSE: ", loss.item())
                print("Epoch ", t, "Test MSE: ", loss_test.item())
        hist[t] = loss.item()
        hist_test[t] = loss_test.item()
        
        if early_stopper.early_stop(loss_test):          
            print("Hit Early Stop")
            break

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 20)
        # Update parameters
        optimiser.step()
        
    return(optimiser, hist, hist_test, y_train_pred)
        
    