import sys
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import random
from common_tune_benchmark import *
from sklearn.model_selection import train_test_split

def getFineTunedModel_glue(model, tokenizer):
    # Get fine-tuning dataset
    train_set, test_set = getBinaryDataset_SST2(tokenizer)

    # Define trainer with default arguments
    training_args = getDefaultTrainingArguments()
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=test_set,)

    # Fine-tune the model
    trainer.train()
    fine_tuned_model = trainer.model
    return fine_tuned_model