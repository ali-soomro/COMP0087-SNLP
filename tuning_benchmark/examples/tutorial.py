from common_tune_benchmark import *
from finance_fine_tune import *
from imdb_tune import *

# *********************************************************************
# *                                                                   *
# *    NOTE: THIS FILE IS NOT SUPPOSED TO RUN                         *
# *                                                                   *
# *********************************************************************

# How to retrieve pre-trained model and tokenizer for DistilBert, which we will be fine-tuning on:
model, tokenizer = getModel_Binary_DistilBert()

# How to fine-tune a model on a particular dataset:
# Format: getFineTunedModel_$DATASET_NAME$(model, tokenizer) <- Returns a model fine tuned on the data set specified 
# Examples below:
fine_tuned_model = getFineTunedModel_Finance(model, tokenizer)
fine_tuned_model = getFineTunedModel_IMDB(model, tokenizer)

# How to EVALUATE a (typically fine-tuned) model on a particular dataset:
# Format: evaluate_model(fine_tuned_model, tokenizer, $STRING_DATASET_NAME$) <- Evaluates model and logs results
# Examples below:
# Return code -1 indicates dataset not found, 0 is success
retcode = evaluate_model(fine_tuned_model, tokenizer, "imdb")
retcode = evaluate_model(fine_tuned_model, tokenizer, "finance")
retcode = evaluate_model(fine_tuned_model, tokenizer, "finance", "Custom Message For Logs") # You can add a custom message as the last parameter

# *********************************************************************
# *                                                                   *
# *    NOTE: THIS FILE IS NOT SUPPOSED TO RUN                         *
# *                                                                   *
# *********************************************************************