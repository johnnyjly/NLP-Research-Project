# =============================== #
#  Naive RNN class baseline model #
# =============================== #
# Use feature based bert or word2Vec to perform naive RNN
# In theory, they can share a same model class.

import torch
import os, bert
import pandas as pd
import numpy as np
import torch.nn as nn

class salaryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(salaryRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # TODO
        assert False, "Implement"