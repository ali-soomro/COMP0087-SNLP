import torch
import sys
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Amazon Product Reviews dataset
dataset = load_dataset('amazon_polarity')

# Define models and tokenizers
models = {
    "bert-base-uncased": (BertForSequenceClassification, BertTokenizer),
    "roberta-base": (RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert-base-uncased": (DistilBertForSequenceClassification, DistilBertTokenizer)
}

# Training arguments
training_args = TrainingArguments(
    output_dir='./output',  # Specify the output directory
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    report_to='none'
)

# Define evaluation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Train and evaluate models
for model_name, (model_class, tokenizer_class) in models.items():
    print(f"Training and evaluating {model_name}...")
    print(device)
    sys.exit(0)
    model = model_class.from_pretrained(model_name, num_labels=2).to(device)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    # Tokenize and preprocess the dataset
    train_texts = dataset['train']['content']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['content']
    test_labels = dataset['test']['label']

    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(train_labels)
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(test_labels)
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        device=device
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Test Accuracy for {model_name}: {eval_result['eval_accuracy']}")