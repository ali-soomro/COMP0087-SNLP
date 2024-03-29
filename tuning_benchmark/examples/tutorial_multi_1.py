from multi_benchmark import *

# *********************************************************************
# *                                                                   *
# *    NOTE: THIS FILE IS NOT SUPPOSED TO RUN                         *
# *                                                                   *
# *********************************************************************

# How to run a particular combination of experiments on datasets defined below:

# DEFINITION OF DATASETS
# 1 = finance
# 2 = imdb'
# 3 = amazon
# 4 = sst2
# DEFINITION OF DATASETS

# ---------------------------------------------------------------------
# EXAMPLE USAGE 1
combination = "124"

# This will take a pretrained Distilbert, fine-tune it on finance (1), imdb (2), and sst2 (4) in that order.
# It will then evaluate it on sst2 (3) and log the required metrics
leave_one_out_fine_tuning_and_evaluation(input_string=combination, model_name=distilbert)
# EXAMPLE USAGE 1
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# EXAMPLE USAGE 2
# Please ensure you have looked at example usage 1 first

combination = "123"  # If "123" then fine-tunes on 1, 2, 3 and tests on 4
# We use assert_valid_string() to ensure that the string is a subset of {1, 2, 3, 4} of size 3 with unique characters only
# We iterate through the list of models in benchmark_all_models() and call leave_one_out_fine_tuning_and_evaluation() inside benchmark_all_models()
benchmark_all_models(combination)
# EXAMPLE USAGE 2
# ---------------------------------------------------------------------

# *********************************************************************
# *                                                                   *
# *    NOTE: THIS FILE IS NOT SUPPOSED TO RUN                         *
# *                                                                   *
# *********************************************************************