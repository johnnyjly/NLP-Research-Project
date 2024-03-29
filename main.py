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

from preprocess import preprocess_data

def plot_loss():
    # TODO: Plot loss and accuracy respect to epoch here
    assert False, 'plot_loss function not implemented'


def train(model, train_loader, criterion, optimizer, device):
    # TODO: Implement the training loop here
    assert False, 'train function not implemented'


def test(model, test_loader, criterion, device):
    # TODO: Implement the testing loop here
    assert False, 'test function not implemented'


def main():
    # Get preprocessed data
    data_path = 'data/preprocessed_data.csv'
    data = preprocess_data(data_path)

    # Split train and test data
    # TODO 

    # Training Loop
    # TODO

    # Testing Loop 
    # TODO

    # plot 
    # TODO


if __name__ == '__main__':
    main()