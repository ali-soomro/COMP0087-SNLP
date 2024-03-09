# Import necessary functions from the common_tune_benchmark module
from common_tune_benchmark import *

# Import function to load dataset from the datasets module
from datasets import load_dataset

# Initialize logging and load dataset
start_logging()  # Start logging with default settings
imdb_dataset_name = 'imdb'  # Specify the name of the dataset
imdb_dataset = load_dataset(imdb_dataset_name)  # Load the IMDb dataset

model_name = "distilbert-base-uncased"

# Evaluate distilbert-base-uncased on the loaded dataset
# evaluateBaseModel(move_tensor_to_gpu(imdb_dataset), imdb_dataset_name, "distilbert-base-uncased")

emotion_dataset = load_dataset("emotion")

model_class, tokenizer_class = getBaseModel("distilbert-base-uncased")
model = model_class.from_pretrained(model_name).to("cuda")
# num_labels = len(dataset['train'].features['label'].names)
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")

model = downstreamTrain(dataset=emotion_dataset, model=model)