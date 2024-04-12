# ==================================== #
# Main entry point for the application #
# ==================================== #
import torch
import os, bert
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from preprocess import preprocess_data
from model_bert_featurebase import BertFeature
from model_bert_rnn import BertRNN
from model_RNN import salaryRNN
from model_bert import salaryBERT


def accuracy(model, dataset, max=1000):
    """
    Estimate the accuracy of `model` over the `dataset`.
    Calculate the accuracy as the proportion of intersection/union.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.
        `max`     - The max number of samples to use to estimate
                    model accuracy

    Returns: a floating-point value between 0 and 1.
    """
    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=1,  # use batch size 1 to prevent padding
                            shuffle=True)
    acc = []
    for i, (x, t) in enumerate(dataloader):
        y = model(x)
        lower = max(y[0], t[0])
        upper = min(y[1], t[1])

        if lower <= upper:
            prop = (upper - lower) / (max(y[1], t[1]) - min(y[0], t[0]))
            acc.append(prop)
        else:
            acc.append(0)

        if i >= max:
            break
    return sum(acc)/max


def plot_loss(iters, train_loss, train_acc):
    # TODO: Plot loss and accuracy respect to epoch here
    plt.figure()
    plt.plot(iters[:len(train_loss)], train_loss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.figure()
    plt.plot(iters[:len(train_acc)], train_acc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")


def train(model, train_data, train_loader, criterion, device, epochs, plot_every=50, plot=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    iters, train_loss, train_acc = [], [], []
    iter_count = 0
    try:
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

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta)

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch} : Average Loss {avg_loss}')
        torch.save(model.state_dict(), 'model.pth')
    finally:
        plot_loss(iters, train_loss, train_acc)


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
    encodings = tokenizer(data['string'], padding='max_length', truncation=True, return_tensors='pt')
    encodings['targets'] = torch.tensor([data['target_l'], data['target_u']])
    return encodings
    
    # return tokenizer(data['string'].tolist(), padding='max_length', truncation=True, return_tensors='pt', return_labels=True)
                  
    
def main():
    # Disable parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Get preprocessed data
    data_path = './data/data_cleaned_2021.csv'
    data = preprocess_data(data_path)

    # Split train and test data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenize_dataset = data.apply(
        lambda x: tokenize_data(x, tokenizer),
        axis=1
    )
    train_dataset, test_dataset = train_test_split(tokenize_dataset, test_size=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    #之后可以加上validation，train里面还没有写validation的部分
    
    #hyperparameters
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = compute_loss
    learning_rate = 0.0001
 
    # models 
    model = salaryBERT()
    
    # Training Loop & plot
    train(model, train_dataset, train_loader, criterion, device, epochs)

    # Evalute Loop 
    # TODO
    evalute(model, test_loader, criterion, device)



if __name__ == '__main__':
    main()