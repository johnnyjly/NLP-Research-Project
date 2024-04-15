import torch.nn as nn
from transformers import BertModel

class BertRNN(nn.Module):
    def __init__(self, rnn_hidden_size, num_layers,bidirectional):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn = nn.GRU(self.bert.config.hidden_size, rnn_hidden_size, num_layers, batch_first=True, bidirectional= bidirectional)
        self.fc = nn.Linear(rnn_hidden_size*2 if bidirectional else rnn_hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output, _ = self.rnn(output.last_hidden_state)
        # avgpool or maxpool
        output = self.fc(output[:, -1, :])
        return output