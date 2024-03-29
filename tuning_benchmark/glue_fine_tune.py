from common_tune_benchmark import *
from finance_fine_tune import *
from glue_tune import *

# Begin logging
start_logging()
logging.info("Training on sst2 started")

# Get pretrained model, tokenizer
model, tokenizer = getModel_Binary_DistilBert()

# Fine tune on IMDB
fine_tuned_model = getFineTunedModel_glue(model, tokenizer)
fine_tuned_model = model
logging.info("Training on sst2 finished")

# Evaluate on Finance
evaluate_model(fine_tuned_model, tokenizer, dataset_name="sst2", custom_model_name="Testing on sst2")