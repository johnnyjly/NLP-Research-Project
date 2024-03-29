# ========================= #
#   Bert Fine Tune Model    #
# ========================= #
# Use Bert and an extra layer to perform fine tune

import torch, os, bert
import pandas as pd
import numpy as np
import torch.nn as nn

class salaryBERT(nn.Module):
    def __init__(self, bert, hidden_size, num_classes):
        super(salaryBERT, self).__init__()
        # FIXME: Probably pretty wrong code here, need to fix
        self.bert = bert
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # TODO
        assert False, "Implement"