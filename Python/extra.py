# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:41:34 2023

@author: lawre
"""
# In[12]:





# In[13]:





# ## Preparation

# #### Processing

# In[14]:


# #Train test validation splits
# #X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(ob, y_name, 0.2)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_date(dat, y_name, dte = forecast_start_date)
# y_val_low = y_val
# print("X_train: " + str(X_train.shape) + " " + str(min(X_train.index)) + ">" + str(max(X_train.index)))
# print("X_test: " + str(X_test.shape)+ " " + str(min(X_test.index)) + ">" + str(max(X_test.index)))
# print("X_val: " + str(X_val.shape)+ " " + str(min(X_val.index)) + ">" + str(max(X_val.index)))
# print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
# print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))
# print("y_val: " + str(y_val.shape)+ " " + str(min(y_val.index)) + ">" + str(max(y_val.index)))


# In[15]:


# import datetime
# forecast_start_date + datetime.timedelta(days=1)


# ### Exploration

# Data visualisation check, although in quantile form the data has an odd structure. 

# In[16]:


fig, ax = plt.subplots()
ax.plot(y_train.index, y_train.values)
ax.grid()
plt.xticks(np.arange(0,y_train.shape[0],math.floor(y_train.shape[0]/4)))
plt.show()


# Data against one of the key features yesterdays mean rolled value in quantile form.

# In[17]:


fig, ax = plt.subplots()
ax.scatter(X_train['WB_CRUDE_BRENT'], y_train)
ax.grid()
plt.show()


# In[18]:


import seaborn as sns
# sns.pairplot(ob)


# Look at correlation of the key feature against target 

# In[19]:


# calculate the spearmans's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import spearmanr
# seed random number generator
# calculate spearman's correlation
# corr, _ = spearmanr(X_train['quant_bef_Last_Rolled'], y_train)
# print('Spearmans correlation: %.3f' % corr)


# ### VIF and Correlation Removal

# ### Subsetting

# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html

#  Create a subset of the best features to feed into the networks to speed up training time.

# In[20]:


# models_fwd = pd.DataFrame(columns=["RSS", "model"])

# tic = time.time()
# predictors = []

# for i in range(1,len(X_train.columns)+1):    
#     models_fwd.loc[i] = forward(predictors, X_train, y_train[y_name])
#     predictors = models_fwd.loc[i]["model"].model.exog_names

# toc = time.time()
# print("Total elapsed time:", (toc-tic), "seconds.")


# In[21]:


# d = []
# for i in range(1, len(models_fwd)):
#     d.append({"model" : i, "r_sq": models_fwd.loc[i, "model"].rsquared})
    
# subsets = pd.DataFrame(d)
# subsets["diff"] = subsets["r_sq"].diff()

# pred_subset = min(subsets[subsets["diff"] < 0.0001].index)
# pred_subset


# In[22]:


# plt.plot(subsets["r_sq"], label="r squared")
# plt.legend()
# plt.show()


# In[23]:


# reg_model_low = models_fwd.loc[pred_subset, "model"]
# print(models_fwd.loc[pred_subset, "model"].summary())


# In[24]:


# subset_cols = reg_model_low.params.index.tolist()
# #'quant_bef_low_high_range', taken out due to high VIF
# X_train_subset = X_train[subset_cols]
# X_test_subset = X_test[subset_cols]
# X_val_subset = X_val[subset_cols]
# #Nieve Model
# # y_pred_reg_low = reg_model_low.predict(X_val_subset) 
# # y_pred_reg_low


# Recheck on the VIF

# At this point the X_train_subset or the orginal X_train can be fed into the tensors depending on how one wants to train the networks

# In[25]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train_arr = scaler.fit_transform(X_train_subset)
# X_val_arr = scaler.transform(X_val_subset)
# X_test_arr = scaler.transform(X_test_subset)

# y_train_arr = scaler.fit_transform(y_train)
# y_val_arr = scaler.transform(y_val)
# y_test_arr = scaler.transform(y_test)

# # make training and test sets in torch
# X_train_subset_tensor = torch.from_numpy(X_train_arr).type(torch.Tensor)
# X_test_subset_tensor = torch.from_numpy(X_test_arr).type(torch.Tensor)
# X_val_subset_tensor = torch.from_numpy(X_val_arr).type(torch.Tensor)
# y_train_tensor = torch.from_numpy(y_train_arr).type(torch.Tensor)
# y_test_tensor = torch.from_numpy(y_test_arr).type(torch.Tensor)
# y_val_tensor = torch.from_numpy(y_val_arr).type(torch.Tensor)

# X_train_subset_tensor = X_train_subset_tensor[:,:,None]
# #y_train_tensor = y_train_tensor[:,None]
# X_test_subset_tensor = X_test_subset_tensor[:,:,None]
# #y_test_tensor = y_test_tensor[:,None]
# X_val_subset_tensor = X_val_subset_tensor[:,:,None]
# #y_val_tensor = y_val_tensor[:,None]

# print('x_train.shape = ',X_train_subset_tensor.shape)
# print('y_train.shape = ',y_train_tensor.shape)
# print('x_test.shape = ',X_test_subset_tensor.shape)
# print('y_test.shape = ',y_test_tensor.shape)
# print('x_val.shape = ',X_val_subset_tensor.shape)
# print('y_val.shape = ',y_val_tensor.shape)


# Trains the regression model

# In[26]:


# y_pred_train_reg_low = reg_model_low.predict(X_train_arr) 
# y_pred_test_reg_low = reg_model_low.predict(X_test_arr) 
# y_pred_val_reg_low = reg_model_low.predict(X_val_arr) 

# trainScore = math.sqrt(mean_squared_error(y_train, y_pred_train_reg_low))
# print('Regression Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(y_test, y_pred_test_reg_low))
# print('Regression Test Score: %.2f RMSE' % (testScore))
# valScore = math.sqrt(mean_squared_error(y_val, y_pred_val_reg_low))
# print('Regression Validation Score: %.2f RMSE' % (valScore))


# ## Modelling

# ## Hyperparameter Optimisation

# https://discuss.pytorch.org/t/hyperparameter-using-skorch-how-to-set-some-optimizer-as-parameter/38057

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# https://skorch.readthedocs.io/en/stable/regressor.html

# This creates two set of parameters. The outer parameters are fed into the LSTM creation and the inner paramterers are fed into the neural net regressor. Items are added to each of the lists and run for the number of required epochs. Althought it doesnt not produce the best model the epochs are held to 100 to save training time. 
# In[62]:




    
#Split data
#Train test validation splits
dat = dat.copy().drop(y_remove, axis=1)#.astype('float64')
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_date(df = dat, target_col = y_name, dte = dte,
                                                                          val_days = val_days, test_days = val_days)

if verbose == True:
    print(dte)
    print("X_train: " + str(X_train.shape) + " " + str(min(X_train.index)) + ">" + str(max(X_train.index)))
    print("X_test: " + str(X_test.shape)+ " " + str(min(X_test.index)) + ">" + str(max(X_test.index)))
    print("X_val: " + str(X_val.shape)+ " " + str(min(X_val.index)) + ">" + str(max(X_val.index)))
    print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
    print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))
    print("y_val: " + str(y_val.shape)+ " " + str(min(y_val.index)) + ">" + str(max(y_val.index)))

#Subset data 
X_train_vif, X_test_vif, X_val_vif = vif_correlation_subset(X_train, X_test, X_val, 
                                                            read_vif_values = read_vif_values, verbose = verbose)

X_train_subset, X_test_subset, X_val_subset, reg_model, subsets = fwd_subset(X_train_vif, X_test_vif, X_val_vif, 
                                                         y_train, y_name, verbose = verbose, 
                                                                    read_subset_values = read_subset_values)

#Create tensors from subset of data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train_subset)
X_val_arr = scaler.transform(X_val_subset)
X_test_arr = scaler.transform(X_test_subset)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

y_pred_train_reg = reg_model.predict(X_train_arr) 
y_pred_test_reg = reg_model.predict(X_test_arr) 
y_pred_val_reg = reg_model.predict(X_val_arr) 

import math
from sklearn.metrics import mean_squared_error
trainScoreReg = math.sqrt(mean_squared_error(y_train, y_pred_train_reg))
testScoreReg = math.sqrt(mean_squared_error(y_test, y_pred_test_reg))
valScoreReg = math.sqrt(mean_squared_error(y_val, y_pred_val_reg))

if verbose == True:
    print('Regression Train Score: %.2f RMSE' % (trainScoreReg))
    print('Regression Test Score: %.2f RMSE' % (testScoreReg))
    print('Regression Validation Score: %.2f RMSE' % (valScoreReg))

    plt.plot(subsets["r_sq"], label="r squared")
    plt.legend()
    plt.show()

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
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
elif model_name == "rnn":
    model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                      layer_dim=num_layers, dropout_prob = dropout)

model = prune_model_l1_unstructured(model, nn.Conv2d, prune_prop)

loss_fn = torch.nn.L1Loss()

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if verbose == True:
    print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

model = prune_model_l1_unstructured(model, nn.Conv2d, prune_prop)

# Train model
#####################
hist = np.zeros(num_epochs)
hist_test = np.zeros(num_epochs)
look_back = 10 # choose sequence length

# Number of steps to unroll
seq_dim = look_back-1  

start_time = time.time() 
for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()

    # Forward pass
    y_train_pred = model(X_train_subset_tensor)
    y_test_pred = model(X_test_subset_tensor)

    loss = loss_fn(y_train_pred, y_train_tensor)
    loss_test = loss_fn(y_test_pred, y_test_tensor)
    if verbose == True:
        if t % 50 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
            print("Epoch ", t, "Test MSE: ", loss_test.item())
    hist[t] = loss.item()
    hist_test[t] = loss_test.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

if verbose == True:
    print("--- %s seconds ---" % (time.time() - start_time))

if verbose == True:
    plt.plot(hist, label="Training loss")
    plt.plot(hist_test, label="Test loss")
    plt.legend()
    plt.title(str(model) + " Loss")
    plt.savefig('lstm_loss_low.png')
    plt.show()

# make test predictions
y_test_pred = model(X_test_subset_tensor)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train_det = scaler.inverse_transform(y_train_tensor.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test_det = scaler.inverse_transform(y_test_tensor.detach().numpy())

# make validation predictions
y_val_pred = model(X_val_subset_tensor)

# invert predictions
y_val_pred = scaler.inverse_transform(y_val_pred.detach().numpy())
y_val_det = scaler.inverse_transform(y_val_tensor.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_det[:,0], y_train_pred[:,0]))
testScore = math.sqrt(mean_squared_error(y_test_det[:,0], y_test_pred[:,0]))
valScore = math.sqrt(mean_squared_error(y_val_det[:,0], y_val_pred[:,0]))

if verbose == True:
    print('Test Score: %.2f RMSE' % (testScore))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Validation Score: %.2f RMSE' % (valScore))

    # Visualising the results
    figure, axes = plt.subplots()
    axes.xaxis_date()

    axes.plot(pd.to_datetime(y_val.index), y_val.values, color = 'red', label = 'Real')
    axes.plot(pd.to_datetime(y_val.index), y_val_pred, color = 'blue', label = 'Predicted')
    plt.title(str(model) + ' Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('lstm.png')
    plt.show()

err_row = pd.DataFrame(data = {"reg_train":trainScoreReg,"reg_test":testScoreReg, "reg_val":valScoreReg,
      "net_train":trainScore,"net_test":testScore, "net_val":valScore#,
        #"hist":hist,"hist_test":hist_test,"y_test_pred":y_test_pred, "y_val_pred":y_val_pred
                              }, index = [str(dte)])

fcst = pd.DataFrame(data = {"dte" : y_val.index, "step": [step] * len(y_val_pred), "y_pred": y_val_pred, "y_val": y_val.values})

return(hist, hist_test, y_test_pred, y_val_pred, err_row, fcst)

# In[103]:


fcst.index = fcst["step"].astype(str) + "_" + fcst["dte"]
fcst


# In[87]:


y_val.values.flatten()




# In[46]:


X_train_vif, X_test_vif, X_val_vif = vif_correlation_subset(X_train, X_test, X_val, read_vif_values = False, verbose = True)


# In[47]:


X_train_subset, X_test_subset, X_val, reg_model = fwd_subset(X_train_vif, X_test_vif, X_val_vif, 
                                                             y_train, y_name, verbose = True)


def train_net(
            dat,
            dte,
            y_name,
            y_remove,
            step,
            model_name = "lstm",
            input_dim = 1,
            hidden_dim = 32,
            num_layers = 4,
            output_dim = 1,
            learning_rate = 0.1,
            weight_decay = 0.000001,
            dropout = 0.2,
            prune_prop = 0.2,
            num_epochs = 200,
            read_vif_values = False,
            verbose = True,
            read_subset_values = False,
            val_days = 50,
            test_days = 50):
    
    #Split data
    #Train test validation splits
    dat_y = dat.copy().drop(y_remove, axis=1)#.astype('float64')
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_date(df = dat, target_col = y_name, dte = dte,
                                                                              val_days = val_days, test_days = val_days)
    
    if verbose == True:
        print(dte)
        print("X_train: " + str(X_train.shape) + " " + str(min(X_train.index)) + ">" + str(max(X_train.index)))
        print("X_test: " + str(X_test.shape)+ " " + str(min(X_test.index)) + ">" + str(max(X_test.index)))
        print("X_val: " + str(X_val.shape)+ " " + str(min(X_val.index)) + ">" + str(max(X_val.index)))
        print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
        print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))
        print("y_val: " + str(y_val.shape)+ " " + str(min(y_val.index)) + ">" + str(max(y_val.index)))
        
    model = ESRNN(max_epochs=5, 
                    freq_of_test=1, 
                    batch_size=32, 
                    learning_rate=0.02, 
                    per_series_lr_multip=0.5,
                    lr_scheduler_step_size=7, 
                    lr_decay=0.5, 
                    gradient_clipping_threshold=50, 
                    rnn_weight_decay=0.0, 
                    noise_std=0.001, 
                    level_variability_penalty=30, 
                    testing_percentile=50, 
                    training_percentile=50,
                    ensemble=True, 
                    max_periods=371, 
                    seasonality=[24, 168], 
                    input_size=24, 
                    output_size=48,
                    cell_type='LSTM', 
                    state_hsize=40, 
                    dilations=[[1, 4, 24, 168]], 
                    add_nl_layer=False,
                    random_seed=1, 
                    device='cpu')

    #Subset data 
    X_train_vif, X_test_vif, X_val_vif = vif_correlation_subset(X_train, X_test, X_val, 
                                                                read_vif_values = read_vif_values, verbose = verbose)
    
    X_train_subset, X_test_subset, X_val_subset, reg_model, subsets = fwd_subset(X_train_vif, X_test_vif, X_val_vif, 
                                                             y_train, y_name, verbose = verbose, 
                                                                        read_subset_values = read_subset_values)

    #Create tensors from subset of data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_arr = scaler.fit_transform(X_train_subset)
    X_val_arr = scaler.transform(X_val_subset)
    X_test_arr = scaler.transform(X_test_subset)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)
    
    y_pred_train_reg = reg_model.predict(X_train_arr) 
    y_pred_test_reg = reg_model.predict(X_test_arr) 
    y_pred_val_reg = reg_model.predict(X_val_arr) 
    
    import math
    from sklearn.metrics import mean_squared_error
    trainScoreReg = math.sqrt(mean_squared_error(y_train, y_pred_train_reg))
    testScoreReg = math.sqrt(mean_squared_error(y_test, y_pred_test_reg))
    valScoreReg = math.sqrt(mean_squared_error(y_val, y_pred_val_reg))
    
    if verbose == True:
        print('Regression Train Score: %.2f RMSE' % (trainScoreReg))
        print('Regression Test Score: %.2f RMSE' % (testScoreReg))
        print('Regression Validation Score: %.2f RMSE' % (valScoreReg))
        
        plt.plot(subsets["r_sq"], label="r squared")
        plt.legend()
        plt.show()

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
        model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    elif model_name == "rnn":
        model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                          layer_dim=num_layers, dropout_prob = dropout)

    model = prune_model_l1_unstructured(model, nn.Conv2d, prune_prop)

    loss_fn = torch.nn.L1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if verbose == True:
        print(model)
        print(len(list(model.parameters())))
        for i in range(len(list(model.parameters()))):
            print(list(model.parameters())[i].size())
        
    model = prune_model_l1_unstructured(model, nn.Conv2d, prune_prop)

    # Train model
    #####################
    hist = np.zeros(num_epochs)
    hist_test = np.zeros(num_epochs)
    look_back = 10 # choose sequence length

    # Number of steps to unroll
    seq_dim = look_back-1  

    start_time = time.time() 
    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(X_train_subset_tensor)
        y_test_pred = model(X_test_subset_tensor)

        loss = loss_fn(y_train_pred, y_train_tensor)
        loss_test = loss_fn(y_test_pred, y_test_tensor)
        if verbose == True:
            if t % 50 == 0 and t !=0:
                print("Epoch ", t, "MSE: ", loss.item())
                print("Epoch ", t, "Test MSE: ", loss_test.item())
        hist[t] = loss.item()
        hist_test[t] = loss_test.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
    
    if verbose == True:
        print("--- %s seconds ---" % (time.time() - start_time))
     
    if verbose == True:
        plt.plot(hist, label="Training loss")
        plt.plot(hist_test, label="Test loss")
        plt.legend()
        plt.title(str(model) + " Loss")
        plt.savefig('lstm_loss_low.png')
        plt.show()

    # make test predictions
    y_test_pred = model(X_test_subset_tensor)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train_det = scaler.inverse_transform(y_train_tensor.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test_det = scaler.inverse_transform(y_test_tensor.detach().numpy())

    # make validation predictions
    y_val_pred = model(X_val_subset_tensor)

    # invert predictions
    y_val_pred = scaler.inverse_transform(y_val_pred.detach().numpy())
    y_val_det = scaler.inverse_transform(y_val_tensor.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train_det[:,0], y_train_pred[:,0]))
    testScore = math.sqrt(mean_squared_error(y_test_det[:,0], y_test_pred[:,0]))
    valScore = math.sqrt(mean_squared_error(y_val_det[:,0], y_val_pred[:,0]))
    
    if verbose == True:
        print('Test Score: %.2f RMSE' % (testScore))
        print('Train Score: %.2f RMSE' % (trainScore))
        print('Validation Score: %.2f RMSE' % (valScore))
        
        # Visualising the results
        figure, axes = plt.subplots()
        axes.xaxis_date()

        axes.plot(pd.to_datetime(y_val.index), y_val.values, color = 'red', label = 'Real')
        axes.plot(pd.to_datetime(y_val.index), y_val_pred, color = 'blue', label = 'Predicted')
        plt.title(str(model) + ' Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('lstm.png')
        plt.show()

    err_row = pd.DataFrame(data = {"reg_train":trainScoreReg,"reg_test":testScoreReg, "reg_val":valScoreReg,
          "net_train":trainScore,"net_test":testScore, "net_val":valScore#,
            #"hist":hist,"hist_test":hist_test,"y_test_pred":y_test_pred, "y_val_pred":y_val_pred
                                  }, index = [str(dte)])
    
    fcst = pd.DataFrame(data = {"dte" : y_val.index.to_list(), "step":[step] * len(y_val_pred), 
                                "y_pred": y_val_pred.flatten(), "y_val": y_val.values.flatten()})

    fcst.index = fcst["step"].astype(str) + "_" + fcst["dte"]

    return(hist, hist_test, y_test_pred, y_val_pred, err_row, fcst)


# ## ESRNN2

# In[51]:





# ## Testing

# In[32]:


# dat,
# dte,
# y_name,
# y_remove,
step = 1
model_name = "lstm"
input_dim = 1
hidden_dim = 32
num_layers = 4
output_dim = 1
learning_rate = 0.1
weight_decay = 0.000001
dropout = 0.2
prune_prop = 0.2
num_epochs = 200
read_vif_values = False
verbose = True
read_subset_values = False
val_days = 50
test_days = 50
y_name = "tgt_crd"
y_remove = "tgt_iron"
dates = dat.iloc[:-100]
dates = dates[(dates.index >= str(forecast_start_date))].index
model_name = "lstm"
dte = dte = datetime.strptime(dates[1], "%Y-%m-%d")
val_days = 50
test_days = 50


# In[33]:


X_df = pd.read_csv(root + "\\data\\X_df.csv")
y_df = pd.read_csv(root + "\\data\\y_df.csv")


# In[42]:


def train_val_test_split_date_esrnn(df, dte, val_days = 50, test_days = 50):
    val_date = dte + timedelta(days=val_days)
    test_max_date = val_date + timedelta(days=test_days)
    df["ds"] = pd.to_datetime(df["ds"])
    train = df[df["ds"] <= str(dte)]
    val = df[(df["ds"] > str(dte)) & (df["ds"] <= str(val_date))]
    test = df[(df["ds"] > str(val_date)) & (df["ds"] <= str(test_max_date))]
    

    return train, val, test

X_train, X_val, X_test = train_val_test_split_date_esrnn(X_df, dte)
y_train, y_val, y_test = train_val_test_split_date_esrnn(y_df, dte)


# In[44]:


# dat_y = dat.copy().drop(y_remove, axis=1)#.astype('float64')
# X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_date(df = dat, target_col = y_name, dte = dte,
#                                                                           val_days = val_days, test_days = val_days)

# if verbose == True:
#     print(dte)
#     print("X_train: " + str(X_train.shape) + " " + str(min(X_train.index)) + ">" + str(max(X_train.index)))
#     print("X_test: " + str(X_test.shape)+ " " + str(min(X_test.index)) + ">" + str(max(X_test.index)))
#     print("X_val: " + str(X_val.shape)+ " " + str(min(X_val.index)) + ">" + str(max(X_val.index)))
#     print("y_train: " + str(y_train.shape)+ " " + str(min(y_train.index)) + ">" + str(max(y_train.index)))
#     print("y_test: " + str(y_test.shape)+ " " + str(min(y_test.index)) + ">" + str(max(y_test.index)))
#     print("y_val: " + str(y_val.shape)+ " " + str(min(y_val.index)) + ">" + str(max(y_val.index)))

model = ESRNN(max_epochs=5, 
                freq_of_test=1, 
                batch_size=32, 
                learning_rate=0.02, 
                per_series_lr_multip=0.5,
                lr_scheduler_step_size=7, 
                lr_decay=0.5, 
                gradient_clipping_threshold=50, 
                rnn_weight_decay=0.0, 
                noise_std=0.001, 
                level_variability_penalty=30, 
                testing_percentile=50, 
                training_percentile=50,
                ensemble=True, 
                max_periods=371, 
                seasonality=[24, 168], 
                input_size=24, 
                output_size=48,
                cell_type='LSTM', 
                state_hsize=40, 
                dilations=[[1, 4, 24, 168]], 
                add_nl_layer=False,
                random_seed=1, 
                device='cpu')

model.fit(X_df = X_train, y_df = y_train, X_test_df = X_test, y_test_df = y_test)


# In[47]:


# Predict on test set
y_hat_df = model.predict(X_test)

# Evaluate predictions
final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train,
                                                             X_test, y_test,
                                                             naive2_seasonality=1)


# In[35]:


X_train_long = pd.melt(X_train.reset_index(), id_vars = "dte", var_name = "unique_id", value_name = "x")
X_train_long


# In[39]:


from ESRNN.m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa

from ESRNN import ESRNN

X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name='Yearly',
                                                               directory = root + '/data',
                                                               num_obs=1000)
X_train_df


# In[40]:


y_train_df


# In[26]:


y_train


