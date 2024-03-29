import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AutoTokenizer
from transformers import AdamW
import logging
import datetime
from typing import Tuple, Dict, Union
import sys
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import random
from common_tune_benchmark import *
from sklearn.model_selection import train_test_split

from evaluate_batch import *

def evaluate_model(fine_tuned_model, tokenizer, dataset_name, custom_model_name='distilbert-base-uncased'):
    retcode = 0
    if dataset_name == 'imdb':
        dataset = load_dataset(dataset_name)  # Load the IMDb dataset
        imdb_test_dataset = dataset['test']
        accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_imdb(fine_tuned_model, tokenizer, move_tensor_to_gpu(imdb_test_dataset))
        printOrLogEvaluationScores(custom_model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)
    elif dataset_name == 'finance':
        train_set, finance_test_set = getBinaryDataset_Financial(tokenizer)
        accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_finance(fine_tuned_model, tokenizer, finance_test_set)
        printOrLogEvaluationScores(custom_model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)
    elif dataset_name == 'amazon':
        train_set, test_set = getReducedTrainTestDataset_Amazon(tokenizer, 0.01)
        accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_amazon(fine_tuned_model, tokenizer, test_set)
        printOrLogEvaluationScores(custom_model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)        
    elif dataset_name == 'sst2':
        train_set, test_set = getBinaryDataset_SST2(tokenizer)
        accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_sst2(fine_tuned_model, tokenizer, test_set)
        printOrLogEvaluationScores(custom_model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)        
    else:
        print("Dataset not found")
        retcode = -1
    return retcode

def getDefaultTrainingArguments():
    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=500,
        save_steps=1000,
        num_train_epochs=3,
        logging_dir="./logs",
        output_dir="./results"
    )
    return training_args

def getBinaryDataset_SST2(tokenizer: PreTrainedTokenizer):
    def tokenize_function(examples):
        # Adjust the field name if necessary. For SST-2, it's typically 'sentence'.
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)
    
    # Load the SST-2 dataset from the GLUE benchmark
    dataset = load_dataset("glue", "sst2")

    # The SST-2 dataset comes with predefined splits
    train_set = dataset['train']
    test_set = dataset['validation']  # GLUE's SST-2 uses 'validation' as the test set
    
    # Tokenize the datasets
    train_set = train_set.map(tokenize_function, batched=True)
    test_set = test_set.map(tokenize_function, batched=True)
    
    return train_set, test_set

def getBinaryDataset_IMDB(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    # Load the IMDB dataset
    dataset = load_dataset("imdb")

    # The IMDB dataset already comes with predefined splits, so you can directly access them
    train_set = dataset['train']
    test_set = dataset['test']

    # Tokenize the datasets
    train_set = train_set.map(tokenize_function, batched=True)
    test_set = test_set.map(tokenize_function, batched=True)
    
    return train_set, test_set

def getBinaryDataset_Amazon(tokenizer: PreTrainedTokenizer):
    # Load the dataset
    dataset = load_dataset("amazon_polarity")

    # Tokenize the text function
    def tokenize_function(examples):
        return tokenizer(examples['content'], padding="max_length", truncation=True)
    
    # Apply the tokenization to the text of the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Split the dataset into training and test sets
    train_set = tokenized_datasets["train"]
    test_set = tokenized_datasets["test"]
    
    print(f"Size of training set: {train_set.num_rows}")
    print(f"Size of test set: {test_set.num_rows}")
    
    return train_set, test_set

def getReducedTrainTestDataset_Amazon(tokenizer: PreTrainedTokenizer, sample_fraction: float = 0.1):
    # Load the dataset
    dataset = load_dataset("amazon_polarity")
    
    # Shuffle and reduce each of the original splits separately
    reduced_train = dataset["train"].shuffle(seed=42).select(range(int(dataset["train"].num_rows * sample_fraction)))
    reduced_test = dataset["test"].shuffle(seed=42).select(range(int(dataset["test"].num_rows * sample_fraction)))
    
    # Tokenize the text function
    def tokenize_function(examples):
        return tokenizer(examples['content'], padding="max_length", truncation=True)
    
    # Apply tokenization to reduced splits
    tokenized_train_set = reduced_train.map(tokenize_function, batched=True)
    tokenized_test_set = reduced_test.map(tokenize_function, batched=True)
    
    return tokenized_train_set, tokenized_test_set
    

def getBinaryDataset_Financial(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)

    # Should be called by filter_dataset
    def adjust_labels(dataset):
        if dataset['label'] == 2:
            dataset['label'] = 1  # Convert positive sentiment to binary label 1
        return dataset

    def filter_and_adjust_dataset(dataset):
        # Filter out neutral sentiments
        filtered_dataset = dataset.filter(lambda example: example['label'] != 1)
        # Adjust labels from 2 to 1
        adjusted_dataset = filtered_dataset.map(adjust_labels)
        return adjusted_dataset
    
    # Load dataset and then filter it
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")['train']

    # Filter dataset
    dataset = filter_and_adjust_dataset(dataset)
    print(dataset)

    # Should be binary labels {0, 1} now after dataset is filtered, removing neutral
    unique_labels = set(dataset['label'])
    print(f"Unique labels in the filtered dataset: {unique_labels}")

    # Split the dataset into train and test sets
    train_test_split = dataset.train_test_split(test_size=0.2)  # For example, 80% train, 20% test

    # Extract the training and test sets
    train_set = train_test_split['train']
    test_set = train_test_split['test']

    # Check the size of each set to confirm the split
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Tokenized datasets after splitting
    train_set = train_set.map(tokenize_function, batched=True)
    test_set = test_set.map(tokenize_function, batched=True)
    
    return train_set, test_set
    

def getModel_Binary_DistilBert():
    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Define model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    return model, tokenizer


def start_logging(log_file=f'evaluation_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log', log_level=logging.INFO):
    """
    Initializes logging configuration.

    Args:
        log_file (str): Path to the log file. If None, a default filename with timestamp will be used.
        log_level (int): Logging level (default: logging.INFO).
    """
    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging started")

def move_tensor_to_gpu(dataset: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Moves tensors in the dataset to the GPU.

    Args:
        dataset (dict): The dataset containing tensors to move.

    Returns:
        dict: The modified dataset with tensors moved to the GPU.
    """
    # Iterate through each key in the dataset
    for key in dataset.keys():
        # Check if the value corresponding to the key is a torch.Tensor
        if isinstance(dataset[key], torch.Tensor):
            # Move the tensor to the GPU
            dataset[key] = dataset[key].cuda()
    return dataset



def getBaseModel(model_name: str) -> Tuple[type, type]:
    """
    Returns the model and tokenizer corresponding to the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing the model class and tokenizer class.  
    """
    models = {
        "bert-base-uncased": (BertForSequenceClassification, BertTokenizer),
        "roberta-base": (RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert-base-uncased": (DistilBertForSequenceClassification, DistilBertTokenizer)
    }
    
    return models.get(model_name)
    

def printOrLogEvaluationScores(model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc, isPrinted=True, isLogged=True):
    """
    Prints or logs evaluation scores for a model.

    Args:
        model_name (str): The name of the model being evaluated.
        accuracy (float): The accuracy score.
        macro_f1 (float): The macro F1 score.
        micro_f1 (float): The micro F1 score.
        weighted_f1 (float): The weighted F1 score.
        mcc (float): The Matthews Correlation Coefficient.
        kappa (float): Cohen's Kappa score.
        roc_auc (float): The ROC AUC score.
        prc_auc (float): The PRC AUC score.
        isPrinted (bool): Whether the scores should be printed (default: True).
        isLogged (bool): Whether the scores should be logged (default: True).
    """
    if isPrinted:
        print(f"Evaluation scores for {model_name}:")
        print(f"Accuracy: {accuracy}")
        print(f"Macro F1 Score: {macro_f1}")
        print(f"Micro F1 Score: {micro_f1}")
        print(f"Weighted F1 Score: {weighted_f1}")
        print(f"Matthews Correlation Coefficient: {mcc}")
        print(f"Cohen's Kappa Score: {kappa}")
        print(f"ROC AUC: {roc_auc}")
        print(f"PRC AUC: {prc_auc}")

    if isLogged:
        logging.info(f"Evaluating {model_name}...")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Macro F1 Score: {macro_f1}")
        logging.info(f"Micro F1 Score: {micro_f1}")
        logging.info(f"Weighted F1 Score: {weighted_f1}")
        logging.info(f"Matthews Correlation Coefficient: {mcc}")
        logging.info(f"Cohen's Kappa Score: {kappa}")
        logging.info(f"ROC AUC: {roc_auc}")
        logging.info(f"PRC AUC: {prc_auc}")

def downstreamTrain(dataset, model, epochs=3):
    # Split dataset
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Define model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Training Loss: {average_train_loss}")
        
    return model


def evaluateModel(model, model_name: str, dataset: Dict[str, Union[torch.Tensor, DataLoader]]):
    print(f"Evaluating {model_name}...")

def evaluateBaseModel(dataset: Dict[str, Union[torch.Tensor, DataLoader]], dataset_name: str, model_name: str) -> None:
    """
    Evaluate a model on a given dataset.

    Args:
        dataset (dict): A dictionary containing train, validation, and test splits of the dataset.
        dataset_name (str): The name of the dataset being evaluated.
        model_name (str): The name of the model being evaluated

    Raises:
        ValueError: If an invalid dataset name or invalid model name is provided.

    Returns:
        None
    """
    
    print(f"Evaluating {model_name}...")
    
    if model_name == 'distilbert-base-uncased':
        model_class, tokenizer_class = getBaseModel("distilbert-base-uncased")
    else:
        raise ValueError(f"Invalid dataset name: {model_name}")
        
    model = model_class.from_pretrained(model_name).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")

    # Check if dataset_name is 'imdb' and use appropriate evaluation function
    if dataset_name == 'imdb':
        test_dataset = dataset['test']
        accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_imdb(model, tokenizer, test_dataset)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    printOrLogEvaluationScores(model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)