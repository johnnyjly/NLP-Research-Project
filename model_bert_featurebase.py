import torch
import torch.nn as nn
from transformers import BertModel

class BertFeature(nn.Module):
    def __init__(self, rnn_hidden_size, num_layers,bidirectional):
        super(BertFeature, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.rnn = nn.GRU(self.bert.config.hidden_size, rnn_hidden_size, num_layers, batch_first=True, bidirectional= bidirectional)
        self.fc = nn.Linear(rnn_hidden_size*2 if bidirectional else rnn_hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.hidden_states[-4:]
        token_embeddings = torch.stack(hidden_states, dim=0).mean(dim=0)
        
        output, _ = self.rnn(token_embeddings)
        # avgpool or maxpool
        output = self.fc(output[:, -1, :])
        return output