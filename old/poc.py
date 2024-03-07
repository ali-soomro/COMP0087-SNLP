import torch
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef, cohen_kappa_score
import logging

# Set up logging
logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Logging started")

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move each tensor in the dataset to the GPU
def move_tensor_to_gpu(dataset):
    for key in dataset.keys():
        if isinstance(dataset[key], torch.Tensor):
            dataset[key] = dataset[key].cuda()
            print("yay")
    return dataset

# Load the Amazon Product Reviews dataset
dataset = load_dataset('amazon_polarity')

# Move the dataset to GPU
dataset = move_tensor_to_gpu(dataset)

# Define models and tokenizers
models = {
    "bert-base-uncased": (BertForSequenceClassification, BertTokenizer),
    "roberta-base": (RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert-base-uncased": (DistilBertForSequenceClassification, DistilBertTokenizer)
}

def evaluate_batch(model, tokenizer, eval_dataset, batch_size=8):
    model.eval()
    total_accuracy = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i:i+batch_size]

            input_ids = tokenizer(batch['content'], truncation=True, padding=True, return_tensors="pt").to(device)['input_ids']
            attention_mask = tokenizer(batch['content'], truncation=True, padding=True, return_tensors="pt").to(device)['attention_mask']
            labels = torch.tensor(batch['label']).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1)

            total_accuracy += accuracy_score(labels.cpu(), predicted_labels.cpu()) * len(batch)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted_labels.cpu().tolist())
            total_samples += len(batch)

    accuracy = total_accuracy / total_samples

    # Calculate F1 Score
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Calculate Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(all_labels, all_predictions)

    # Calculate Cohen's Kappa Score
    kappa = cohen_kappa_score(all_labels, all_predictions)

    # Calculate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall Curve (PRC) and AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    prc_auc = auc(recall, precision)

    return accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc

# Evaluate models
for model_name, (model_class, tokenizer_class) in models.items():
    print(f"Evaluating {model_name}...")
    logging.info(f"Evaluating {model_name}...")
    
    # Load pretrained model and tokenizer
    model = model_class.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
    
    # Evaluate on the test dataset
    test_dataset = dataset['test']
    accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch(model, tokenizer, test_dataset)
    
    print(f"Accuracy for {model_name}: {accuracy}")
    print(f"Macro F1 Score: {macro_f1}")
    print(f"Micro F1 Score: {micro_f1}")
    print(f"Weighted F1 Score: {weighted_f1}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"Cohen's Kappa Score: {kappa}")
    print(f"ROC AUC: {roc_auc}")
    print(f"PRC AUC: {prc_auc}")
    
    # Log scores to file
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Macro F1 Score: {macro_f1}")
    logging.info(f"Micro F1 Score: {micro_f1}")
    logging.info(f"Weighted F1 Score: {weighted_f1}")
    logging.info(f"Matthews Correlation Coefficient: {mcc}")
    logging.info(f"Cohen's Kappa Score: {kappa}")
    logging.info(f"ROC AUC: {roc_auc}")
    logging.info(f"PRC AUC: {prc_auc}")