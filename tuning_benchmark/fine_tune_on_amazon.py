from common_tune_benchmark import *
from finance_fine_tune import *
from amazon_tune import *
from imdb_tune import *

# Begin logging
start_logging()
logging.info("Training on Amazon started")

# Get pretrained model, tokenizer
model, tokenizer = getModel_Binary_DistilBert()

# Fine tune on Amazon
# To get base benchmark for Amazon simply comment out the line below
fine_tuned_model = getFineTunedModel_Amazon(model, tokenizer)
fine_tuned_model = model

logging.info("Training on Amazon finished")
evaluate_model(fine_tuned_model, tokenizer, dataset_name="amazon", custom_model_name="Testing on Amazon")