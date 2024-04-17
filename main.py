# ==================================== #
# Main entry point for the application #
# ==================================== #
import torch
import os, argparse, random
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from preprocess import preprocess_data
from model_bert_featurebase import BertFeature
from model_bert_rnn import BertRNN
from model_RNN import salaryRNN
from model_bert import salaryBERT


def accuracy(model, dataset, n_max=1000):
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
    for i, d in enumerate(dataloader):
        x, t = d['input_ids'].squeeze(1), d['targets'].squeeze(0)

        if model.__class__.__name__ == 'salaryRNN':
            y = model(x)
        else:
            attention_mask = d['attention_mask'].squeeze(1)
            y = model(x, attention_mask)

        y = y.squeeze(0)
        lower = max(y[0], t[0])
        upper = min(y[1], t[1])

        if lower <= upper:
            prop = (upper - lower) / (max(y[1], t[1]) - min(y[0], t[0]))
            acc.append(prop)
        else:
            acc.append(0)

        if i >= n_max:
            break
    return sum(acc) / min(n_max, i)


def plot_loss(iters, train_loss, val_loss, val_acc):
    iters = torch.tensor(iters).cpu().numpy()
    train_loss = torch.tensor(train_loss).cpu().numpy()
    val_loss = torch.tensor(val_loss).cpu().numpy()
    val_acc = torch.tensor(val_acc).cpu().numpy()
    plt.figure()
    plt.plot(iters[:len(train_loss)], train_loss, label='Train Loss')
    plt.plot(iters[:len(val_loss)], val_loss, label='Validation Loss')
    # Add legend
    plt.legend()
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("loss.png")


    plt.clf()
    plt.figure()
    plt.plot(iters[:len(val_acc)], val_acc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy.png")


def train(model, train_data, train_loader, val_data, val_loader, criterion, epochs, learning_rate=0.0001, patience=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iters, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    iter_count = 0
    divergence_count = 0
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            # Debugging
            input_ids = batch['input_ids'].squeeze(1)
            targets = batch['targets']

            if model.__class__.__name__ == 'salaryRNN':
                outputs = model(input_ids)
            else:
                attention_mask = batch['attention_mask'].squeeze(1)
                outputs = model(input_ids, attention_mask)

            optimizer.zero_grad()
            loss = compute_loss(criterion, outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            iter_count += 1

        iters.append(iter_count)
        avg_loss = total_loss / len(train_loader)
        train_loss.append(float(avg_loss))
        print(iter_count, "Train Loss:", float(avg_loss))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].squeeze(1)
                targets = batch['targets']

                if model.__class__.__name__ == 'salaryRNN':
                    outputs = model(input_ids)
                else:
                    attention_mask = batch['attention_mask'].squeeze(1)
                    outputs = model(input_ids, attention_mask)

                loss = compute_loss(criterion, outputs, targets)
                total_val_loss += loss.item()

        avg_loss = total_val_loss / len(val_loader)
        val_acc = accuracy(model, val_data)
        print(f'End of Epoch {epoch}, Validation Loss: {avg_loss}, Validation Accuracy: {val_acc}')
        
        
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            no_improvement = 0
            best_model = model.state_dict()
        else:
            no_improvement += 1
            if no_improvement > patience:
                print(f"Early stopping at epoch {epoch} with best validation loss {best_val_loss}")
                model.load_state_dict(best_model)
                break
            
    torch.save(model.state_dict(), 'model.pth')
    return iters, train_loss, val_loss, val_acc


def compute_loss(criterion, outputs, targets):
    loss1 = criterion(outputs[:, 0], targets[:, 0])
    loss2 = criterion(outputs[:, 1], targets[:, 1])
    return (loss1 + loss2) / 2


def evaluate(model, test_data, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].squeeze(1)
            targets = batch['targets']

            if model.__class__.__name__ == 'salaryRNN':
                outputs = model(input_ids)
            else:
                attention_mask = batch['attention_mask'].squeeze(1)
                outputs = model(input_ids, attention_mask)

            loss = compute_loss(criterion, outputs, targets)
            print(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    acc = accuracy(model, test_data)
    print(f'Average Loss on Test Set: {avg_loss}, Average Accuracy: {acc}')


def tokenize_data(data, tokenizer, device):
    encodings = tokenizer(data['string'], padding='max_length', truncation=True, return_tensors='pt').to(device)
    encodings['targets'] = torch.FloatTensor([data['target_l'], data['target_u']]).to(device)
    return encodings

    # return tokenizer(data['string'].tolist(), padding='max_length', truncation=True, return_tensors='pt', return_labels=True)


def main(args: argparse.Namespace):
    # TODO: Argparse -Johnny

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

    train_dataset, test_dataset = train_test_split(tokenize_dataset, test_size=0.4)
    test_dataset, val_dataset = train_test_split(test_dataset, test_size=0.5)

    # Reset indices after split
    # train_dataset.reset_index(drop=True, inplace=True)
    # test_dataset.reset_index(drop=True, inplace=True)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 之后可以加上validation，train里面还没有写validation的部分

    # hyperparameters
    epochs = args.epochs
    criterion = torch.nn.MSELoss().to(device)
    learning_rate = args.lr

    # models
    # A dictionary of models to select from
    models = {
        'BertFeature': "BertFeature(512, 2, True)",
        'BertRNN': "BertRNN(512, 2, True)",
        'salaryRNN': "salaryRNN(tokenizer.vocab_size, 300, 512, 1, 2, True)",
        'salaryBERT': "salaryBERT()"
    }
    model = eval(models[args.model])
    if args.model != 'salaryRNN':
        for param in model.bert.parameters():
            param.requires_grad = False
    model.to(device)

    print("Start Training")
    # Training Loop & plot
    iters, train_loss, val_loss, val_acc = train(model, train_dataset, train_loader,val_dataset, val_loader, criterion,
                                                            epochs, learning_rate=learning_rate)
    plot_loss(iters, train_loss, train_acc, val_loss, val_acc)

    # Evaluate Loop
    # TODO
    print("Start Evaluation")
    evaluate(model, test_dataset, test_loader, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/data_cleaned_2021.csv')
    parser.add_argument("--model", type=str, help='BertFeature, BertRNN, salaryRNN, salaryBERT', required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5, help='learning rate')
    parser.add_argument("--seed", type=int, default=413, help='random seed')

    arguments = parser.parse_args()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    main(arguments)
