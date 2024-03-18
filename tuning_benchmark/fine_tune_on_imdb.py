from common_tune_benchmark import *
from finance_fine_tune import *
from imdb_tune import *

# Begin logging
start_logging()
logging.info("Training on IMDB started")

# Get pretrained model, tokenizer
model, tokenizer = getModel_Binary_DistilBert()

# Fine tune on finance
# fine_tuned_model = getFineTunedModel_IMDB(model, tokenizer)
fine_tuned_model = model

logging.info("Training on IMDB finished")

# Evaluate on IMDB
evaluate_model(fine_tuned_model, tokenizer, dataset_name="finance", custom_model_name="Fine-tuned on IMDB - Testing on Finance")