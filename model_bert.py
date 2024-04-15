# ========================= #
#   Bert Fine Tune Model    #
# ========================= #
# Use Bert and an extra layer to perform fine tune
import torch.nn as nn
from transformers import BertModel

class salaryBERT(nn.Module):
    def __init__(self):
        super(salaryBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.fc(output.pooler_output)
        return output