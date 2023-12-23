# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:21:41 2023

@author: lawre
"""
from torch import nn
import torch
from lstm import LSTM
from es_rnn.DRNN import DRNN
import math
from rnn import RNN
from gru import GRU
# Build ESRNN model
#####################
# Here we define our model as a class
#Key components:
        # Deseasonalisation and resseasonalisation
        # Ensamble with RNN stack to give average of forecast
        # Normalisation
        # Forecasting of all series using the deseasonalised data 
#Issues
    #The methods wa deisgned for the M4 only and we have more info about the frequency
# Notes
    # Added alpha base smoothing parameter, lower number the more smoothed 
    
# Inputs
# es_type the type of smoothing that is applied 
# 1 = Level only with alpha smoothing
# 2 = Lev and Trend. Uses alpha and beta smoothing
class ESRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, num_series, 
                 dilations,
                 seasonality = 7, alpha_smoothing = 0.5, beta_smoothing = 0.1,
                 state_hsize = 50,
                 no_trend = True, 
                 input_window_size = 7, output_window_size = 50,
                 rnn_cell_type = "LSTM", add_nl_layer = True,
                 multiplicative_seasonality = True,
                 output_prediction = False):
        super(ESRNN, self).__init__()
        
        init_lev_sms = []
        init_seas_sms = []
        init_seasonalities = []
        
        #For each series, sets up a tensor of 1 to start off the seasonality tensors. 
        #0.5 is midway in the sigmoid 
        for i in range(num_series):
            init_lev_sms.append(nn.Parameter(torch.Tensor([alpha_smoothing]), requires_grad=True))
            init_seas_sms.append(nn.Parameter(torch.Tensor([alpha_smoothing]), requires_grad=True))
            init_seasonalities.append(nn.Parameter((torch.ones(seasonality) * alpha_smoothing), requires_grad=True))

        #Exponential Smoothing Parameters#
        self.no_trend = no_trend
        self.beta_smoothing = beta_smoothing

        #Creates the parameter list that can be used later in the model run
        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        self.init_seas_sms = nn.ParameterList(init_seas_sms)
        self.init_seasonalities = nn.ParameterList(init_seasonalities)
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
        
        # Seasonality
        self.seasonality = seasonality

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)

        # Readout layer
        self.fc = nn.Linear(num_series, output_dim)
        
        # Number of hidden layers
        self.output_dim = output_dim
        
        self.input_dim = input_dim
        
        self.relu = nn.ReLU()
        self.act = nn.Tanh()
        
        #Logistic Sigmoid to be used in the seasonality decomp
        self.logistic = nn.Sigmoid()
        
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        
        #The dilations are how many time steps to skip when running the RNN
        #It ingests as a seasonality and then outputs as the output window
        self.resid_drnn = ResidualDRNN(dilations = dilations, cell_type = rnn_cell_type,
                                       input_window_size = input_window_size, state_hsize = state_hsize)
        
        self.nl_layer = nn.Linear(state_hsize, state_hsize)
        
        self.scoring = nn.Linear(state_hsize, output_window_size)
        
        self.add_nl_layer = add_nl_layer
        
        self.multiplicative_seasonality = multiplicative_seasonality
        
        self.output_prediction = output_prediction

    def forward(self, x, level_variability_penalty = 80, ):
        
        
        self.train()
        
        #Transpose x to match the M4 format
        xt = torch.transpose(x, 0, 1).reshape((x.shape[1],x.shape[0]))
        
        #idxs is the batch index
        #Lev sms are the smoothing levels or alpha. Make the data between 0  and 1
        #These are explicity loaded with the number of series so are expecting that as the input
        lev_sms = self.logistic(torch.stack([self.init_lev_sms[idx] for idx in range(x.shape[1])]).squeeze(1))
        seas_sms = self.logistic(torch.stack([self.init_seas_sms[idx] for idx in range(x.shape[1])]).squeeze(1))
        #For each series stack into 
        init_seasonalities = torch.stack([self.init_seasonalities[idx] for idx in range(x.shape[1])])

        seasonalities = []
        # PRIME SEASONALITY
        # Creates tensor for each seasonality length
        # Stacks them on top of each other and puts it into a list
        # Exponential to reduce to 0 and 1
        #13 vectors of the seasonality values
        for i in range(self.seasonality):
            seasonalities.append(torch.exp(init_seasonalities[:, i]))
        seasonalities.append(torch.exp(init_seasonalities[:, 0]))

        #Create the ES levels and applies first one
        #Different to the second as it doesn't apply the es formula
        levs = []
        beta_levs = []
        levs.append(xt[:, 0] / seasonalities[0])
        beta_levs.append(xt[:, 0] / seasonalities[0])
        
        log_diff_of_levels = []
        
        #For the number of series that are present deseasonalise by the seasonality level that is whon above 
        #Take the series, 
        # =============================================================================
        #         for k in range(1, xt.shape[1]):
        #             # CALCULATE LEVEL FOR CURRENT TIMESTEP TO NORMALIZE RNN
        #             new_lev = lev_sms * (xt[:, k] / seasonalities[k]) + (1 - lev_sms) * levs[k - 1]
        #             levs.append(new_lev)
        # 
        #             # STORE DIFFERENCE TO PENALIZE LATER
        #             log_diff_of_levels.append(torch.log(new_lev / levs[k - 1]))
        # 
        #             # CALCULATE SEASONALITY TO DESEASONALIZE THE DATA FOR RNN
        #             seasonalities.append(seas_sms * (xt[:, k] / new_lev) + (1 - seas_sms) * seasonalities[k])
        # =============================================================================
        
        # =============================================================================
        # Code changes 
        # I changed the single loop to double loop.
        # It was clear that for each series you deseaonalise the data and then move onto the next
        # However the seasonality made above creates a tensor for each seasonality, not season
        # This means 
        # Changed as the tensor that was coming out the other end was different dimensions to each 
        # other when I tried to stack them
        # I dont know how they got that to work as the seasonality and series points are 
        # Different
        # =============================================================================
        
        # =============================================================================
        # Exponential Smoothing Details
        # The paper remvoes the linear trend as the NN was tasked with a non linear trend. 
        # Put it into hyper opt
        # Added linear trend analysis back in as this can be just as useful. The hyper opt can choose
        # Beta smoothing should be low otherwise it overfits https://otexts.com/fpp3/holt.html
        # =============================================================================
        
        #seasonalities2 = []
        #Gets the seasonalities plus the 12 extra months
        for series in range(1, xt.shape[1]):
        #for season in range(self.seasonality):
            # CALCULATE LEVEL FOR CURRENT TIMESTEP TO NORMALIZE RNN
            #Applies the exponential smoothing from previous level

            # deseasonalise step by step for each time step based on previous step 
            if self.no_trend:
                #https://otexts.com/fpp3/ses.html component form
                #lt =     α          yt                                    + (1 − α)lt−1
                new_lev = lev_sms * (xt[:, series] / seasonalities[series]) + (1 - lev_sms) * levs[series - 1]#season second
                #this the new set of smoothed data for each series and seasonal period.
                
                levs.append(new_lev)

                # STORE DIFFERENCE TO -1668PENALIZE LATER
                log_diff_of_levels.append(torch.log(new_lev / levs[series - 1]))

                # CALCULATE SEASONALITY TO DESEASONALIZE THE DATA FOR RNN
                
                #st+K = βyt /lt + (1 − β)st
                #https://otexts.com/fpp3/holt-winters.html multiplicative 
                #seasonalities series is seaonality - 1 
                #This doesnt include the linear trend as pect of the equation
                if self.multiplicative_seasonality:
                    seasonalities.append(seas_sms * (xt[:, series] / new_lev) + (1 - seas_sms) * seasonalities[series])#season last
                else:
                    seasonalities.append(seas_sms * (xt[:, series] - new_lev) + (1 - seas_sms) * seasonalities[series])#season last
                
                
            else:
                #Does the level equation then adds the trend equation
                new_lev = lev_sms * (xt[:, series] / seasonalities[series]) + (1 - lev_sms) * levs[series - 1]#season second
                
                #https://otexts.com/fpp3/holt.html
                #βt = β(st – st-1) + (1 – β)bt-1
                beta_level = (self.beta_smoothing * (new_lev - levs[series - 1])) + ((1 - self.beta_smoothing) * beta_levs[series - 1])
                beta_levs.append(beta_level)
                new_lev = lev_sms * (xt[:, series] / seasonalities[series]) + (1 - lev_sms) * (levs[series - 1] + beta_level)#season second
                
                levs.append(new_lev)

                # STORE DIFFERENCE TO -1668PENALIZE LATER
                log_diff_of_levels.append(torch.log(new_lev / levs[series - 1]))
                #Added beta into equation
                
                if self.multiplicative_seasonality:
                    seasonalities.append(seas_sms * (xt[:, series] / (new_lev + beta_level)) + (1 - seas_sms) * seasonalities[series])#season last
                else:
                    seasonalities.append(seas_sms * (xt[:, series] - new_lev - beta_level) + (1 - seas_sms) * seasonalities[series])#season last
            
        #Stack and store the deseasonalised and smoohted data
        #Observations, 
        seasonalities_stacked = torch.stack(seasonalities).transpose(1, 0)
        levs_stacked = torch.stack(levs).transpose(1, 0)


        loss_mean_sq_log_diff_level = 0
        #0 in yearly config so ignore
        #multiplier applied to the level wiggliness penalty. Applies only to the seasonal models.
        #The value is not very sensitive, as changing it even by 50% would not make a big difference. 
        #In the orginal, doersn't actually go anywhere
        if level_variability_penalty > 0: 
            sq_log_diff = torch.stack(
                [(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
            loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)

        #Extend the seasonality out for the forecast period
        if self.output_window_size > self.seasonality:
            start_seasonality_ext = seasonalities_stacked.shape[1] - self.output_window_size + self.seasonality
            # start_seasonality_ext + self.output_window_size - self.seasonality
            end_seasonality_ext = seasonalities_stacked.shape[1] 
            #end_seasonality_ext = start_seasonality_ext + self.output_dim - self.seasonality
            seasonalities_stacked = torch.cat((seasonalities_stacked, 
                                               seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]),
                                              dim=1)

        #For each timestep in the data deseaonalise all series
        window_input_list = []
        window_output_list = []
        for i in range(self.input_window_size - 1, xt.shape[1]):
            #print(i)
            input_window_start = i + 1 - self.input_window_size
            input_window_end = i + 1
            #print("Input" + str(input_window_start) + " " + str(input_window_end))
            
            train_deseas_window_input = xt[:, input_window_start:input_window_end] / \
                seasonalities_stacked[:, input_window_start:input_window_end]
            train_deseas_norm_window_input = (train_deseas_window_input / levs_stacked[:, i].unsqueeze(1))
            #This exends the train window for the dummies if factors are used #last is concat info
            train_deseas_norm_cat_window_input = torch.cat((train_deseas_norm_window_input, torch.ones(xt.shape[0],1)), dim=1)  
            window_input_list.append(train_deseas_norm_cat_window_input)

            output_window_start = i - 1
            
            # =============================================================================
            #             if i >= xt.shape[1] - self.output_window_size:
            #                 output_window_end = xt.shape[1]
            #             else:
            # =============================================================================
            output_window_end = i + 1 + self.output_window_size#self.config['output_size']
            
            if i < xt.shape[1] - self.output_window_size:
                #print("Output" + str(output_window_start) + " " + str(output_window_end))
                train_deseas_window_output = xt[:, output_window_start:output_window_end] / \
                                             seasonalities_stacked[:, output_window_start:output_window_end]
                train_deseas_norm_window_output = (train_deseas_window_output / levs_stacked[:, i].unsqueeze(1))
                
                window_output_list.append(train_deseas_norm_window_output)
            
            # =============================================================================
            #             if i < xt.shape[1] - self.output_window_size:#self.config['output_size']:
            #                 train_deseas_window_output = xt[:, output_window_start:output_window_end] / \
            #                                              seasonalities_stacked[:, output_window_start:output_window_end]
            #                 train_deseas_norm_window_output = (train_deseas_window_output / levs_stacked[:, i].unsqueeze(1))
            #                 window_output_list.append(train_deseas_norm_window_output)
            # =============================================================================

        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)

        #Run the stacked LSTM's

        #self.train()
        #TODO: this could be swapped
        window_input_0 = torch.nan_to_num(window_input, nan=0.0)
        # removed as using whole dataset test separate
        network_pred = self.series_forward(window_input_0)#[:-self.output_window_size])
        network_act = window_output
        
        #Reduces the prediction dimension down to the series and number of observations 
        #out_fcst_mean = torch.mean(network_pred, dim=2, keepdim=False)
        #Amend 20/12/2023
# =============================================================================
#         out_fcst_mean_relu = self.relu(network_act)
#         
#         out_fcst_mean = out_fcst_mean_relu[:,:,:].reshape((out_fcst_mean_relu.shape[0],out_fcst_mean_relu.shape[2],out_fcst_mean_relu.shape[1]))
#         # Check for NaN and infinite values
#         nan_mask = torch.isnan(out_fcst_mean)
#         inf_mask = torch.isinf(out_fcst_mean)
#         
#         # Create a combined mask to identify NaN and infinite values
#         nan_inf_mask = nan_mask | inf_mask
#         
#         # Remove NaN and infinite values from the tensor
#         mean_all = torch.median(out_fcst_mean[~nan_inf_mask])
#         
#         out_fcst_mean_nona = torch.nan_to_num(out_fcst_mean, nan = float(mean_all))
#         out_series_mean_fc = self.fc(out_fcst_mean_nona)#.flatten()
#         out_series_mean_nona = torch.nan_to_num(out_series_mean_fc, nan = float(mean_all))
#         out_series_mean = torch.mean(out_series_mean_nona, dim = 1).flatten()
#         
#         out_pad = torch.nn.functional.pad(out_series_mean, (xt.shape[1] - out_fcst_mean_relu.shape[0], 0), \
#                                           mode = "constant", value = float(out_series_mean[0])).reshape(-1, 1)
# =============================================================================
        
        #Original 20/12/2023
        #Reduceds the prediction dimension down to the series and number of observations 
        #out_fcst_mean = torch.mean(network_pred, dim=2, keepdim=False)
        out_fcst_mean = network_pred[:,:,1]
        #out_fcst_mean_unwind = torch.nan_to_num(torch.exp(out_fcst_mean * levs_stacked[:, i].unsqueeze(1) * seasonalities_stacked[:, output_window_start:output_window_end]))
        
        #out_series_mean = torch.nanmean(out_fcst_mean, dim=1, keepdim=False)
        out_fcst_mean_relu = self.relu(out_fcst_mean)
        out_series_mean = self.fc(out_fcst_mean_relu).flatten()
        
        #print(out_series_mean)
        #Pad seasons at the start of the tensor using the first value and working back wards
        out_pad = torch.nn.functional.pad(out_series_mean, (xt.shape[1] - out_series_mean.shape[0], 0), \
                                          mode = "constant", value = float(out_series_mean[0])).reshape(-1, 1)

        #hold_out_act = test if testing else val

        #hold_out_act_deseas = hold_out_act.float() / seasonalities_stacked[:, -self.config['output_size']:]
        #hold_out_act_deseas_norm = hold_out_act_deseas / levs_stacked[:, -1].unsqueeze(1)
        
# =============================================================================
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
# 
#         # Initialize cell state
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
# 
#         # We need to detach as we are doing truncated backpropagation through time (BPTT)
#         # If we don't, we'll backprop all the way to the start even after going through another batch
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
# 
# 
#         out = self.fc(network_pred[:, -1, :]) 
#         # Index hidden state of last time step
#         # out.size() --> 100, 32, 100
#         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
#         out = self.fc(out[:, -1, :]) 
#         # out.size() --> 100, 10

# =============================================================================

        if self.output_prediction:
            
            self.eval()

            network_output_non_train = self.series_forward(window_input)

            # USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS
            hold_out_output_reseas = network_output_non_train[-1] * seasonalities_stacked[:, -self.output_window_size:]
            hold_out_output_renorm = hold_out_output_reseas * levs_stacked[:, -1].unsqueeze(1)

            hold_out_pred = hold_out_output_renorm * torch.gt(hold_out_output_renorm, 0).float()
            
            hold_out_pred = torch.nan_to_num(hold_out_pred, nan=0.0)
            
            # Replicate the first column to create the padding
            padding = hold_out_pred[:, :1].expand(-1, xt.shape[1] - out_series_mean.shape[0] + 1)
            
            # Concatenate the original tensor with the padding
            hold_out_pred_pad = torch.cat((hold_out_pred, padding), dim=1)
            
            #Flip dimensions so the series are top down rather than across
            hold_out_pred_pad_flip = hold_out_pred_pad.permute(1, 0)
            
            #Add dimension
            hold_out_pred_pad_dim3 = hold_out_pred_pad_flip.unsqueeze(2)
            
            return out_pad, hold_out_pred_pad_dim3
        else:
            return out_pad
    
    def series_forward(self, data):
        data = self.resid_drnn(data)
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.act(data)
        data = self.scoring(data)
        return data


class ResidualDRNN(nn.Module):
    def __init__(self, dilations, input_window_size, state_hsize, num_of_categories = 1, cell_type = "LSTM"):
        super(ResidualDRNN, self).__init__()

        layers = []
        for grp_num in range(len(dilations)):

            if grp_num == 0:
                input_size = input_window_size + num_of_categories
            else:
                input_size = state_hsize

            l = DRNN(n_input = input_size,
                     n_hidden = state_hsize,
                     n_layers = len(dilations[grp_num]),
                     dilations = dilations[grp_num],
                     cell_type = cell_type)

            layers.append(l)

        self.rnn_stack = nn.Sequential(*layers)

    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            out, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                out += residual
            input_data = out
        return out
    
    
    