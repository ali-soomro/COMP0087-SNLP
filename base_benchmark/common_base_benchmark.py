import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AutoTokenizer
import logging
import datetime

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

# Move each tensor in the dataset to the GPU if possible
def move_tensor_to_gpu(dataset):
    for key in dataset.keys():
        if isinstance(dataset[key], torch.Tensor):
            dataset[key] = dataset[key].cuda()
    return dataset

# Define and return models and tokenizers
def getAllModels():
    models = {
        "bert-base-uncased": (BertForSequenceClassification, BertTokenizer),
        "roberta-base": (RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert-base-uncased": (DistilBertForSequenceClassification, DistilBertTokenizer)
    }
    return models

# Main function to call once have loaded the dataset
def evaluateAllModels(dataset, dataset_name):
    # Evaluate models
    for model_name, (model_class, tokenizer_class) in getAllModels().items():
        print(f"Evaluating {model_name}...")
        
        # Load pretrained model and tokenizer
        model = model_class.from_pretrained(model_name).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
        
        # Evaluate on the test dataset
        test_dataset = dataset['test']
        if dataset_name == 'imdb':
            accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_imdb(model, tokenizer, test_dataset)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        print(f"Accuracy for {model_name}: {accuracy}")
        print(f"Macro F1 Score: {macro_f1}")
        print(f"Micro F1 Score: {micro_f1}")
        print(f"Weighted F1 Score: {weighted_f1}")
        print(f"Matthews Correlation Coefficient: {mcc}")
        print(f"Cohen's Kappa Score: {kappa}")
        print(f"ROC AUC: {roc_auc}")
        print(f"PRC AUC: {prc_auc}")
        
        # Log scores to file
        logging.info(f"Evaluating {model_name}...")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Macro F1 Score: {macro_f1}")
        logging.info(f"Micro F1 Score: {micro_f1}")
        logging.info(f"Weighted F1 Score: {weighted_f1}")
        logging.info(f"Matthews Correlation Coefficient: {mcc}")
        logging.info(f"Cohen's Kappa Score: {kappa}")
        logging.info(f"ROC AUC: {roc_auc}")
        logging.info(f"PRC AUC: {prc_auc}")