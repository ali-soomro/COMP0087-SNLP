from common_base_benchmark import *
from datasets import load_dataset

start_logging()
dataset_name = 'imdb'
dataset = load_dataset(dataset_name)
evaluateAllModels(move_tensor_to_gpu(dataset), dataset_name)