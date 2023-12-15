# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:39:25 2023

@author: lawre
"""
import statsmodels.api as sm
import time
import pandas as pd
import numpy as np
#Subset Selection
from numba import jit, cuda

jit(target_backend='cuda') 
def processSubset(feature_set, X, y):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

jit(target_backend='cuda') 
def forward(predictors, X, y):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    
    tic = time.time()
    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p], X, y))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

jit(target_backend='cuda') 
def fwd_subset(X_train, y_train, y_name, dte, xtra_desc, verbose = True, read_subset_values = False, rss_cutoff = 5,
               max_subset_cols = None):
    
    if max_subset_cols == None:
        subset_cols = len(X_train.columns)
    else:
        subset_cols = max_subset_cols
    
    if read_subset_values:
        X_train_subset1 = pd.read_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_X_train_subset.csv", index_col = 0)
        X_train_subset = X_train[X_train_subset1.columns]  
        from statsmodels.regression.linear_model import OLSResults
        reg_model = OLSResults.load("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_reg_model.pickle")
        subsets = pd.read_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_subsets.csv", index_col = 0)

    else:
        models_fwd = pd.DataFrame(columns=["RSS", "model"])

        tic = time.time()
        predictors = []

        for i in range(1,subset_cols+1):    
            models_fwd.loc[i] = forward(predictors, X_train, y_train[y_name])
            predictors = models_fwd.loc[i]["model"].model.exog_names

        toc = time.time()

        if verbose:
            print("Total elapsed time:", (toc-tic), "seconds.")

        d = []
        for i in range(1, len(models_fwd)):
            d.append({"model" : i, "r_sq": models_fwd.loc[i, "model"].rsquared,
                      "rss": models_fwd.loc[i, "RSS"]})

        subsets = pd.DataFrame(d)
        subsets["diff"] = subsets["rss"].diff().abs()
        subsets = subsets.dropna()

        pred_subset = min(subsets[subsets["diff"] < rss_cutoff].index)
        pred_subset

        reg_model = models_fwd.loc[pred_subset, "model"]
        if verbose:
            print(reg_model.summary())

        subset_cols = reg_model.params.index.tolist()
        #'quant_bef_low_high_range', taken out due to high VIF
        X_train_subset = X_train[subset_cols]
        
        reg_model.save("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_reg_model.pickle")
        X_train_subset.head(5).to_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc  + "_X_train_subset.csv")
        subsets.to_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc +  "_subsets.csv")
        #models_fwd.to_csv("models_fwd.csv")
    
    return(X_train_subset, reg_model, subsets)

jit(target_backend='cuda') 
def vif_correlation_subset(X_train, dte, xtra_desc, read_vif_values = False, verbose = True):

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if read_vif_values:
        X_train_vif = pd.read_csv("data/" + dte.strftime("%Y%m%d") +xtra_desc + "_vif.csv", index_col = 0)
        X_train = X_train[X_train_vif.columns]  

    else:  
        # Create correlation matrix
        corr_matrix = X_train.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        # Drop features 
        X_train.drop(to_drop, axis=1, inplace=True)

        #https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

        X_train_vif = X_train
        # vif_cols = list(X_train.columns)
        # vif_values = X_train.values

        for index, col in enumerate(X_train.columns):
            if verbose:
                print(col)
                print(index)

            vif_data = pd.DataFrame()
            vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i)
                              for i in range(len(X_train_vif.columns))]
            vif_data["feature"] = X_train_vif.columns  

            feature = vif_data[vif_data["feature"] == col]
            if verbose:
                print(feature)

            if feature["VIF"].values[0] > 10:
                #vif_cols.remove(feature["feature"].values[0])  
                X_train = X_train.drop([feature["feature"].values[0]], axis=1)
                if verbose:
                    print("Removing " + feature["feature"].values[0])
                    #print(X_Train)

            X_train.head(5).to_csv("data/" + xtra_desc + dte.strftime("%Y%m%d") + "_vif.csv")
 
    
    return(X_train)

def corr_cutoff(X_train, y_train, dte, xtra_desc, read_corr_values = False, verbose = True, cutoff = 0.4):
 #https://www.geeksforgeeks.org/create-a-correlation-matrix-using-python/
    if read_corr_values:
        X_train_vif = pd.read_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc  + "_corr.csv", index_col = 0)
        X_train = X_train[X_train_vif.columns]     
    else:
        
        dat = pd.merge(X_train, y_train, left_index = True, right_index = True)
        
        # form correlation matrix
        matrix = dat.corr('pearson').abs()
        corr_cols = matrix[matrix[y_train.columns[0]] > cutoff].index.delete(-1)
        print("Correlation matrix is : ")
        print(corr_cols)
        
        X_train_corr = X_train[corr_cols]
    
        X_train.head(5).to_csv("data/" + dte.strftime("%Y%m%d") + xtra_desc + "_corr.csv")
        
    return(X_train_corr)