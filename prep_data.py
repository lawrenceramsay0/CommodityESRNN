# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:41:25 2023

@author: lawre
"""
from train_test_split_custom import feature_label_split, train_val_test_split, train_val_test_split_date, train_val_test_split_idx
from subset import processSubset, forward, fwd_subset, vif_correlation_subset, corr_cutoff
import pandas as pd
from predict_features import predict_features

def prep_data(dat, y_name, dte, val_days, test_days, xtra_desc, verbose = True,
              run_corr_cutoff = False, read_corr_values = False,
              run_vif = False, read_vif_values = False,
              run_fwd_selection = False, read_subset_values = False,
              read_pred_values = False, max_subset_cols = None,
              rss_cutoff = 5
              ):
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_idx(df = dat, target_col = y_name, dte = dte,
                                                                              val_days = val_days, test_days = test_days)
    
    #Subset data 
    if run_corr_cutoff:
        X_train = corr_cutoff(X_train, y_train, read_corr_values = read_corr_values, verbose = verbose, dte = dte, 
                              xtra_desc = xtra_desc)
        
        X_test = X_test[X_train.columns]        
        X_val = X_val[X_train.columns] 
    
    if run_vif:
        X_train = vif_correlation_subset(X_train, read_vif_values = read_vif_values, verbose = verbose, dte = dte, 
                              xtra_desc = xtra_desc)
        
        X_test = X_test[X_train.columns]        
        X_val = X_val[X_train.columns] 

    if run_fwd_selection:
        X_train, reg_model, subsets = fwd_subset(X_train, y_train, y_name, verbose = verbose, dte = dte,
                                                 read_subset_values = read_subset_values, xtra_desc = xtra_desc,
                                                 max_subset_cols = max_subset_cols, rss_cutoff = rss_cutoff)
        
        X_test = X_test[X_train.columns]
        X_val = X_val[X_train.columns]


    if read_pred_values:
        X_test_pred = pd.read_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_X_test_pred.csv", index_col = 0)

    else:
        X_test_pred = predict_features(X_train, X_test, test_days = test_days, verbose = verbose)
        X_test_pred.to_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_X_test_pred.csv")
        
    if verbose == True:
        print(dte)
        print("X_train: " + str(X_train.shape) + " " + str(min(X_train.index)) + ">" + str(max(X_train.index)))
        print("X_test: " + str(X_test.shape)+ " " + str(min(X_test.index)) + ">" + str(max(X_test.index)))
        print("X_val: " + str(X_val.shape)+ " " + str(min(X_val.index)) + ">" + str(max(X_val.index)))
        print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
        print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))
        print("y_val: " + str(y_val.shape)+ " " + str(min(y_val.index)) + ">" + str(max(y_val.index)))
        
    if run_fwd_selection:
        return(X_train, X_val, X_test_pred, y_train, y_val, y_test, reg_model, subsets)
    else:
        return(X_train, X_val, X_test_pred, y_train, y_val, y_test)