# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 13:20:37 2023

@author: lawre
"""

# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

#Train Test Split
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_val_test_split_date(df, target_col, dte, val_days = 50, test_days = 50):
    test_date = dte + timedelta(days=val_days)
    val_max_date = test_date + timedelta(days=test_days)

    X_train = df[df.index <= str(dte)]
    X_test = df[(df.index > str(dte)) & (df.index <= str(test_date))]
    X_val = df[(df.index > str(test_date)) & (df.index <= str(val_max_date))]

    X_train, y_train = feature_label_split(X_train, target_col)
    X_val, y_val = feature_label_split(X_val, target_col)
    X_test, y_test = feature_label_split(X_test, target_col)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_test_split_date(df, target_col, dte, test_days = 50):
    test_max_date = dte + timedelta(days=test_days)

    X_train = df[df.index <= str(dte)]
    X_test = df[(df.index > str(dte)) & (df.index <= str(test_max_date))]

    X_train, y_train = feature_label_split(X_train, target_col)
    X_test, y_test = feature_label_split(X_test, target_col)
    
    return X_train, X_test, y_train, y_test

def train_val_test_split_idx(df, target_col, dte, val_days = 50, test_days = 50):
    test_date = dte + timedelta(days=int(val_days))
    val_max_date = test_date + timedelta(days=int(test_days))

    X_train = df[df.index <= str(dte)]
    X_test_val = df[(df.index > str(dte))].iloc[0:(val_days + test_days)]
    X_test = X_test_val.iloc[0:test_days]
    X_val = X_test_val.iloc[test_days:(val_days + test_days + 1)]

    X_train, y_train = feature_label_split(X_train, target_col)
    X_val, y_val = feature_label_split(X_val, target_col)
    X_test, y_test = feature_label_split(X_test, target_col)
    
    return X_train, X_val, X_test, y_train, y_val, y_test