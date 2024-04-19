# CSC413Project

The course project for CSC413 2024S

## Project Target

- Create an NLP model such that user enter job description and exerience in
  natural language, and output range of salary (Classification)

## Project Setup
```bash
[windows]
pip install -r requirements.txt

[linux/mac]
pip3 install -r requirements.txt
```

## Running The Project
The model will automatically use CUDA environment if available.
### Training a new model
```
python3 main.py 
    --model {one of BertRNN/BertFeature/salaryRNN/salaryBERT} 
    --lr {optional learning rate, default=2e-5} 
	--epochs {optional max epoch, default=500} 
	--patience {optional patience until early stopping, default=10}
```
After the training, a `model.pth` will be saved, in which we store the trained
weights of the new model.

### Predict salaries using existing model
```
python3 main.py 
	--model {one of BertRNN/BertFeature/salaryRNN/salaryBERT} 
	--predict {txt file where we have the input job description} 
	--use_trained {model state_dict file, need to match with model parameter}
```
This will load the trained model parameters and predict a salary range for 
the job description provided. In our project, the model state dicts are stored
in `.pth` files. We have included the trained params of a BertRNN model `BertRNN_model_84.pth` in our repo.


## Links

#### Dataset:

- https://www.kaggle.com/datasets/nikhilbhathi/data-scientist-salary-us-glassdoor

#### How-to's

- how to fine tune using bert:
  https://huggingface.co/docs/transformers/training

## Ideas

- Use bert as pre-trained model, train with training dataset descriptions.
- Discard job-category | company name; Keep job-description | job-requirements
  | experience category

## Steps

### Preprocessing Data

- Remove unnecessary columns.
- Re-categorize experience and salary.
- Tokenize Natural Language inputs.

### Model Building

- Concatenate "experience", "job_location", "job_desig", "key_skills",
  "job_description"
- Use "Salary" as target.
- Fine-Tune BERT by adding a linear layer and train the last epoch with our
  training data and target.
- Use Bert Pre-trained model
- No decoder, directly output two logits representing upper and lower bound

### Compare With Naive RNN

- Hand code RNN, 2 version encoding
  - Word2Vec version
  - BERT feature version
- Compare output of same test inputs
