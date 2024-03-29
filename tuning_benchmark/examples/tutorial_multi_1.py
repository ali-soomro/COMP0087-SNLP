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

# EXAMPLE USAGE
combination = "124"

# This will take a pretrained Distilbert, fine-tune it on finance (1), imdb (2), and sst2 (4) in that order.
# It will then evaluate it on sst2 (3) and log the required metrics
leave_one_out_fine_tuning_and_evaluation(combination)

# EXAMPLE USAGE

# *********************************************************************
# *                                                                   *
# *    NOTE: THIS FILE IS NOT SUPPOSED TO RUN                         *
# *                                                                   *
# *********************************************************************