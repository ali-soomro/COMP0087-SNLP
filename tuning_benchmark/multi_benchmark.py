from common_tune_benchmark import *
from finance_fine_tune import *
from imdb_tune import *

# Begin logging
start_logging()
logging.info("Multi fine tuning started")

# Get pretrained model, tokenizer
model, tokenizer = getModel_Binary_DistilBert()

# 1 = Finance
# 2 = IMDB
# 3 = Amazon
# 4 = sst2
# The fine tuning order
given_string = '124'

# Get testing dataset
full_set = {'1', '2', '3', '4'}
given_set = set(given_string)
missing_numbers = full_set - given_set
testing_string_int = missing_numbers.pop() 

fine_tuned_model = model
if testing_string_int == 1:
    evaluate_model(fine_tuned_model, tokenizer, dataset_name="finance", custom_model_name="Evaluating on Finance using"+given_string)
elif testing_string_int == 2:
    evaluate_model(fine_tuned_model, tokenizer, dataset_name="imdb", custom_model_name="Evaluating on IMDB using"+given_string)
elif testing_string_int == 3:
    print("hi")