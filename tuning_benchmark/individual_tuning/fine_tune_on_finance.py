from common_tune_benchmark import *
from finance_fine_tune import *

# Begin logging
start_logging()
logging.info("Training on Finance started")

# Get pretrained model, tokenizer
model, tokenizer = getModel_Binary_DistilBert()

# Fine tune on finance
fine_tuned_model = getFineTunedModel_Finance(model, tokenizer)
logging.info("Training on Finance finished")

# Evaluate on IMDB
evaluate_model(fine_tuned_model, tokenizer, dataset_name="imdb", custom_model_name="Fine-tuned on Finance - Testing on IMDB")