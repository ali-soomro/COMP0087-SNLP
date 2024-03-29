# COMP0087 - Statistical Natural Language Processing


This project evaluates different Language Model-based approaches for sentiment analysis. The goal is to explore how Large Language Models (LLMs) pre-trained on generic text data can be effectively applied to sentiment analysis tasks across various domains. These models are then fine-tuned using Transfer Learning techniques and the findings are reported. This project also investigates the robustness of these models to variations in sentiment expression.

## Table of Contents

1. [Research Questions](#research-questions)
2. [Prerequisites](#prerequisites)
3. [Usage](#usage)
4. [Project Structure](#project-structure)

## Research Questions

### 1. Transfer Learning
- **How effectively do Language Model architectures pre-trained on generic text data transfer their knowledge to sentiment analysis tasks in specific domains (e.g., product reviews, movie reviews, social media)?**
- **Can Language Model architectures adapt to sentiment analysis tasks in new domains with minimal fine-tuning or domain-specific data?**
- **Are there any transfer learning techniques which are particularly effective for sentiment analysis?**

### 2. Robustness
- **How robust are Language Model architectures to variations in sentiment expression, such as negations, sarcasm, or context-dependent sentiment?**

## Prerequisites

The experiments are performed on UCL's Computer Science "blaze" machine with Python 3.8.12

1. Clone the repository:

   ```bash
   git clone https://github.com/ali-soomro/COMP0087-SNLP.git
   
2. Ensure that Python 3.8.12 is installed, and then install the following pip packages in your Python environment (I used a virtual environment):

   ```bash
   pip install -r requirements.txt

## Usage

## Project Structure
