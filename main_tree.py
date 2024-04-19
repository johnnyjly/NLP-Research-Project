# ==================================== #
# Main entry point for the application #
# ==================================== #
import torch
import numpy as np
import os, argparse, random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from transformers import AutoTokenizer

from preprocess import preprocess_data

def compute_loss(criterion, outputs, targets):
    loss1 = criterion(outputs[:, 0], targets[:, 0])
    loss2 = criterion(outputs[:, 1], targets[:, 1])
    return (loss1 + loss2) / 2

def tokenize_data(data, tokenizer, device):
    encodings = tokenizer(data['string'], padding='max_length', truncation=True, return_tensors='pt').to(device)
    encodings['targets'] = torch.FloatTensor([data['target_l'], data['target_u']]).to(device)
    return encodings

    # return tokenizer(data['string'].tolist(), padding='max_length', truncation=True, return_tensors='pt', return_labels=True)


def main(args: argparse.Namespace):

    # Disable parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current Device: {}".format(device))

    # Get preprocessed data
    data_path = args.data_path
    data = preprocess_data(data_path)

    # Split train and test data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenize_dataset = data.apply(
        lambda x: tokenize_data(x, tokenizer, device),
        axis=1
    )

    tokenize_dataset = tokenize_dataset.tolist()

    train_dataset, test_dataset = train_test_split(tokenize_dataset, test_size=0.2)
    # test_dataset, val_dataset = train_test_split(test_dataset, test_size=0.5)

    # Reset indices after split
    # train_dataset.reset_index(drop=True, inplace=True)
    # test_dataset.reset_index(drop=True, inplace=True)

    X_train = [data['input_ids'].squeeze(0).numpy() for data in train_dataset]
    y_train = np.array([data['targets'].squeeze(0).numpy()  for data in train_dataset])
    
    print(X_train[0].shape)
    print(y_train[0].shape)
    
    X_test = [data['input_ids'].squeeze(0).numpy()  for data in test_dataset]
    y_test = np.array([data['targets'].squeeze(0).numpy()  for data in test_dataset])
    
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.3, max_depth=3)
    multi_output_model = MultiOutputRegressor(model)
    multi_output_model.fit(X_train, y_train)
    
    predictions = multi_output_model.predict(X_test)
    print(predictions.shape)
    # Calculate acc
    acc = []
    for i, (y, t) in enumerate(zip(predictions, y_test)):
        lower = max(y[0], t[0])
        upper = min(y[1], t[1])

        if lower <= upper:
            prop = (upper - lower) / (max(y[1], t[1]) - min(y[0], t[0]))
            acc.append(prop)
        else:
            acc.append(0)

    print(f'acc is {sum(acc) / i}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/data_cleaned_2021.csv')
    # parser.add_argument("--model", type=str, help='BertFeature, BertRNN, salaryRNN, salaryBERT', required=True)
    # Hyperparameters
    parser.add_argument("--seed", type=int, default=413, help='random seed')
    
    arguments = parser.parse_args()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    main(arguments)
