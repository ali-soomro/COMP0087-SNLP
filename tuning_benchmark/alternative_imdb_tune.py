import sys
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import random
from common_tune_benchmark import *
from sklearn.model_selection import train_test_split


start_logging()

model, tokenizer = getModel_Binary_DistilBert()
train_set, test_set = getBinaryDataset_Financial(tokenizer)
training_args = getDefaultTrainingArguments()

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
)

# Fine-tune the model
trainer.train()

# Retrieve the fine-tuned model
fine_tuned_model = trainer.model

# Evaluate on the test dataset
dataset_name = 'imdb'  # Specify the name of the dataset
dataset = load_dataset(dataset_name)  # Load the IMDb dataset
imdb_test_dataset = dataset['test']

accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc = evaluate_batch_imdb(fine_tuned_model, tokenizer, imdb_test_dataset)

printOrLogEvaluationScores('distilbert-base-uncased', accuracy, macro_f1, micro_f1, weighted_f1, mcc, kappa, roc_auc, prc_auc)