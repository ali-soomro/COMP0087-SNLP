from common_tune_benchmark import *
from common_fine_tuning_training import *

start_logging()
model, tokenizer = getModel_Binary_DistilBert()
evaluate_model(model, tokenizer, dataset_name='adverserial', customMessage="Testing")