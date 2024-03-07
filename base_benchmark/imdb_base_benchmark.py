# Import necessary functions from the common_base_benchmark module
from common_base_benchmark import *

# Import function to load dataset from the datasets module
from datasets import load_dataset

# Initialize logging and load dataset
start_logging()  # Start logging with default settings
dataset_name = 'imdb'  # Specify the name of the dataset
dataset = load_dataset(dataset_name)  # Load the IMDb dataset

# Evaluate all models on the loaded dataset
evaluateAllModels(move_tensor_to_gpu(dataset), dataset_name)