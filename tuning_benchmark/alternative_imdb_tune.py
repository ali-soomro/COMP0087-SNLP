import sys
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import random
from common_tune_benchmark import *
from finance_fine_tune import *
from sklearn.model_selection import train_test_split

# Begin logging
start_logging()

# Get pretrained model, tokenizer
model, tokenizer = getModel_Binary_DistilBert()

# Fine tune on finance
fine_tuned_model = getFineTunedModel_Finance(model, tokenizer)

# Evaluate on IMDB
evaluate_model(fine_tuned_model, tokenizer, "imdb")