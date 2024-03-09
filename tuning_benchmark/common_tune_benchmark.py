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

from evaluate_batch import *

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
    # Split the dataset
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    # Prepare DataLoader
    batch_size = 16  # Adjust according to your resources
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define the model architecture
    num_labels = len(dataset['train'].features['label'].names)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)  # Assuming 'input_ids' is generated by tokenizer
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