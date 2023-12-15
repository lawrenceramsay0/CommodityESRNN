# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:36:23 2023

@author: lawre
"""
from torch import nn
import torch
# Build LSTM model
#####################
# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=False)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=False)

# =============================================================================
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
# 
#         # Initialize cell state
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
# =============================================================================

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        #TODO: add relu activation
        out = self.relu(out)
        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out2 = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out2
    