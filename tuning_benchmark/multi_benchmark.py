from common_tune_benchmark import *
from common_fine_tuning_training import *

def leave_one_out_fine_tuning_and_evaluation(input_string, model_name):
    # Defining fine-tuning functions which take a model and tokenizer, and return the fine-tuned model
    dataset_functions = {
        '1': ('finance', getFineTunedModel_Finance, "Fine tuning on Finance"),
        '2': ('imdb', getFineTunedModel_IMDB, "Fine tuning on IMDB"),
        '3': ('amazon', getFineTunedModel_Amazon, "Fine tuning on Amazon"),
        '4': ('sst2', getFineTunedModel_glue, "Fine tuning on sst2")
    }

    datasets_to_use = list(dataset_functions.keys())

    # Determine which datasets to fine-tune on and which one to test on
    fine_tuning_datasets = [d for d in input_string]
    testing_dataset = list(set(datasets_to_use) - set(fine_tuning_datasets))[0]  # Assuming only one left out
    
    
    # Load the original pre-trained model and tokenizer
    if model_name == "Distilbert":
        model, tokenizer = getModel_Binary_DistilBert()

    for dataset_key in fine_tuning_datasets:
        dataset_name, fine_tuning_function, _ = dataset_functions[dataset_key]
        logging.info("Fine-tuning on " + dataset_name)
        # Fine-tuning the model on the particular dataset
        model = fine_tuning_function(model, tokenizer)
    
    # Evaluate on the left-out dataset
    test_dataset_name, _, custom_model_name = dataset_functions[testing_dataset]
    logMessage = "Evaluating on " + test_dataset_name + " using model: " + model_name
    evaluate_model(model, tokenizer, dataset_name=test_dataset_name, customMessage=logMessage)

combination = "124"
start_logging(combination=combination)
leave_one_out_fine_tuning_and_evaluation(combination, model_name="Distilbert")  # Fine-tunes on 1, 2, 3 and tests on 4