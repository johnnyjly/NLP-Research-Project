# =============================== #
#  Naive RNN class baseline model #
# =============================== #
# Use feature based bert or word2Vec to perform naive RNN
# In theory, they can share a same model class.

import torch
import torch.nn as nn

class salaryRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_classes, bidirectional):
        super(salaryRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        embedded = self.emb(x)
        output, _ = self.rnn(embedded)
        max_output, _ = torch.max(output, dim=1)
        output = self.fc(max_output)
        return output