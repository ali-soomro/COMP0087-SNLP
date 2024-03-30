from common_tune_benchmark import *
from common_fine_tuning_training import *

def benchmark_all_models(input_string):
    model_names = ["distilbert", "roberta", "bert"]
    
    for model_name in model_names:
        print(f"Starting fine-tuning and evaluation for {model_name.upper()} model.")
        logging.info("Starting fine-tuning for " + model_name.upper())
        fine_tuned_model, tokenizer = leave_one_out_fine_tuning_and_evaluation(input_string, model_name=model_name)
        logging.info("Completed fine-tuning for " + model_name.upper())
        evaluate_model(fine_tuned_model, tokenizer, dataset_name='adverserial', customMessage="ADVERSERIAL")
        logging.info("EVALUATION COMPLETED FOR " + model_name.upper())

def leave_one_out_fine_tuning_and_evaluation(input_string, model_name):
    # Defining fine-tuning functions which take a model and tokenizer, and return the fine-tuned model
    dataset_functions = {
        '1': ('finance', getFineTunedModel_Finance, "Fine tuning on Finance started"),
        '2': ('imdb', getFineTunedModel_IMDB, "Fine tuning on IMDB started"),
        '3': ('amazon', getFineTunedModel_Amazon, "Fine tuning on Amazon started"),
        '4': ('sst2', getFineTunedModel_glue, "Fine tuning on sst2 started")
    }

    datasets_to_use = list(dataset_functions.keys())

    # Determine which datasets to fine-tune on and which one to test on
    fine_tuning_datasets = [d for d in input_string]
    testing_dataset = list(set(datasets_to_use) - set(fine_tuning_datasets))[0]  # Assuming only one left out
    
    
    # Load the original pre-trained model and tokenizer
    if model_name == "distilbert":
        model, tokenizer = getModel_Binary_DistilBert()
    elif model_name == "roberta":
        model, tokenizer = getModel_Binary_RoBERTa()
    elif model_name == "bert":
        model, tokenizer = getModel_Binary_BERT()

    for dataset_key in fine_tuning_datasets:
        dataset_name, fine_tuning_function, _ = dataset_functions[dataset_key]
        logging.info("Fine-tuning on " + dataset_name)
        # Fine-tuning the model on the particular dataset
        model = fine_tuning_function(model, tokenizer)
    
    # Evaluate on the left-out dataset
    test_dataset_name, _, custom_model_name = dataset_functions[testing_dataset]
    logMessage = "Evaluating on " + test_dataset_name + " using model: " + model_name
    evaluate_model(model, tokenizer, dataset_name=test_dataset_name, customMessage=logMessage)
    
    return model, tokenizer

def assert_valid_string(input_string):
    # Check if the length of the string is 3
    assert len(input_string) == 3, "The input string must be exactly 3 characters long."
    
    # Check if all characters are unique
    assert len(set(input_string)) == 3, "All characters in the input string must be unique."
    
    # Check if each character is one of "1", "2", "3", or "4"
    valid_chars = set("1234")
    assert set(input_string).issubset(valid_chars), "The input string must only contain characters from '1', '2', '3', and '4'."

try:
    combination = "132"  # If "123" then fine-tunes on 1, 2, 3 and tests on 4
    assert_valid_string(combination)
except AssertionError as e:
    print(f"Please enter a valid combination: {e}")
    start_logging(combination=combination)
    logging.info("Please use a valid combination")
    sys.exit(-1)
start_logging(combination=combination)
# leave_one_out_fine_tuning_and_evaluation(combination, model_name="distilbert")
benchmark_all_models(combination)