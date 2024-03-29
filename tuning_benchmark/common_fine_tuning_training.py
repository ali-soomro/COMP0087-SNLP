from common_tune_benchmark import *

def getFineTunedModel_Amazon(model, tokenizer):
    # Get fine-tuning dataset
    train_set, test_set = getReducedTrainTestDataset_Amazon(tokenizer, sample_fraction=0.01)

    # Define trainer with default arguments
    training_args = getDefaultTrainingArguments()
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=test_set,)

    # Fine-tune the model
    trainer.train()
    fine_tuned_model = trainer.model
    return fine_tuned_model

def getFineTunedModel_Finance(model, tokenizer):
    # Get fine-tuning dataset
    train_set, test_set = getBinaryDataset_Financial(tokenizer)

    # Define trainer with default arguments
    training_args = getDefaultTrainingArguments()
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=test_set,)

    # Fine-tune the model
    trainer.train()
    fine_tuned_model = trainer.model
    return fine_tuned_model

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

def getFineTunedModel_IMDB(model, tokenizer):
    # Get fine-tuning dataset
    train_set, test_set = getBinaryDataset_IMDB(tokenizer)

    # Define trainer with default arguments
    training_args = getDefaultTrainingArguments()
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=test_set,)

    # Fine-tune the model
    trainer.train()
    fine_tuned_model = trainer.model
    return fine_tuned_model