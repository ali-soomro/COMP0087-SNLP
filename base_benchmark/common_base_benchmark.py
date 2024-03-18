import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AutoTokenizer
import logging
import datetime


def start_logging(log_file=f'evaluation_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log', log_level=logging.INFO):
    """
    Initializes logging configuration.

    Args:
        log_file (str): Path to the log file. If None, a default filename with timestamp will be used.
        log_level (int): Logging level (default: logging.INFO).
    """
    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging started")

def move_tensor_to_gpu(dataset):
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


# Define and return models and tokenizers
def getAllModels():
    """
    Retrieves a dictionary of pre-trained models and their corresponding tokenizers.

    Returns:
        dict: A dictionary where keys are model names and values are tuples containing the model class and tokenizer class.
    """
    # Define a dictionary mapping model names to tuples of model class and tokenizer class
    models = {
        "bert-base-uncased": (BertForSequenceClassification, BertTokenizer),
        "roberta-base": (RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert-base-uncased": (DistilBertForSequenceClassification, DistilBertTokenizer)
    }
    return models

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
        logging.info(f"Evaluation scores for {model_name}...")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Macro F1 Score: {macro_f1}")
        logging.info(f"Micro F1 Score: {micro_f1}")
        logging.info(f"Weighted F1 Score: {weighted_f1}")
        logging.info(f"Matthews Correlation Coefficient: {mcc}")
        logging.info(f"Cohen's Kappa Score: {kappa}")
        logging.info(f"ROC AUC: {roc_auc}")
        logging.info(f"PRC AUC: {prc_auc}")

def evaluateAllModels(dataset, dataset_name):
    """
    Evaluate multiple models on a given dataset.

    Args:
        dataset (dict): A dictionary containing train, validation, and test splits of the dataset.
        dataset_name (str): The name of the dataset being evaluated.

    Raises:
        ValueError: If an invalid dataset name is provided.

    Returns:
        None
    """
    # Iterate through each model
    for model_name, (model_class, tokenizer_class) in getAllModels().items():
        print(f"Evaluating {model_name}...")
        
        # Load pretrained model and tokenizer
        model = model_class.from_pretrained(model_name).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
        
        # Evaluate on the test dataset
        test_dataset = dataset['test']
        
        # Check if dataset_name is 'imdb' and use appropriate evaluation function
        if dataset_name == 'imdb':
            accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_imdb(model, tokenizer, test_dataset)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        # Print or log evaluation scores
        printOrLogEvaluationScores(model_name, accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)