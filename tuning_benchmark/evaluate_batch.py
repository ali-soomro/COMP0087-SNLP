import torch
from transformers import PreTrainedTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef, cohen_kappa_score
from torch.nn import Module
from datasets import DatasetDict

def evaluate_batch_imdb(model: Module, tokenizer: PreTrainedTokenizer, eval_dataset: DatasetDict, batch_size: int = 8):
    """
    Evaluate the given model on the IMDb dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for tokenizing input text.
        eval_dataset (Dataset): The dataset to evaluate the model on.
        batch_size (int, optional): Batch size for evaluation. Defaults to 8.

    Returns:
        tuple: A tuple containing evaluation metrics (accuracy, macro_f1, micro_f1, weighted_f1,
        mcc, kappa, roc_auc, prc_auc).
    """
    
    # Set the model to evaluation mode
    
    # Determine the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize variables to store evaluation metrics
    total_accuracy = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []
    
    # Iterate over the evaluation dataset in batches
    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i:i+batch_size]
            # Access text data and labels from the batch
            texts = batch['text']  # Adjust 'text' to the correct key for text data
            labels = torch.tensor(batch['label']).to(device)  # Adjust 'label' to the correct key for labels

            # Tokenize text data
            inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Forward pass through the model
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1)

            # Compute accuracy
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
