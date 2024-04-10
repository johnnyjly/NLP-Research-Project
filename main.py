# ==================================== #
# Main entry point for the application #
# ==================================== #
import torch
import os, bert
import pandas as pd
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from preprocess import preprocess_data


def plot_loss():
    # TODO: Plot loss and accuracy respect to epoch here
    assert False, 'plot_loss function not implemented'


def train(model, train_loader, criterion, device, epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            if model.__class__.__name__ == 'salaryBERT':
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            elif model.__class__.__name__ == 'salaryRNN':
                outputs = model(input_ids)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} : Average Loss {avg_loss}')
    torch.save(model.state_dict(), 'model.pth')

def compute_loss(outputs, targets):
    lower_loss = torch.nn.functional.mse_loss(outputs[:, 0], targets[:, 0])
    upper_loss = torch.nn.functional.mse_loss(outputs[:, 1], targets[:, 1])
    return lower_loss + upper_loss


def evalute(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            if model.__class__.__name__ == 'salaryBERT':
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            elif model.__class__.__name__ == 'salaryRNN':
                outputs = model(input_ids)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Average Loss on Test Set: {avg_loss}')

def tokenize_data(data, tokenizer):
    return tokenizer(data['string'].tolist(), padding='max_length', truncation=True)
    # return tokenizer(data['string'].tolist(), padding='max_length', truncation=True, return_tensors='pt', return_labels=True)
                  
    
def main():
    # Get preprocessed data
    data_path = './data/data_cleaned_2021.csv'
    skill_list, category_list, data = preprocess_data(data_path)

    # Split train and test data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenize_dataset = data.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    train_dataset, test_dataset = train_test_split(tokenize_dataset, test_size=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    #hyperparameters
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = compute_loss
    learning_rate = 0.0001
 
    # models 
    
    # Training Loop
    # TODO
    train(model, train_loader, criterion, device, epochs, learning_rate)

    # Evalute Loop 
    # TODO
    evalute(model, test_loader, criterion, device)

    # plot 
    # TODO


if __name__ == '__main__':
    main()