# Twitter Sentiment Analysis

Perform sentiment analysis on Twitter data using the Naive Bayes classifier and NLTK.



## Introduction

This project uses machine learning techniques to classify tweets as positive or negative. It leverages the NLTK library for natural language processing and the Naive Bayes classifier for sentiment analysis.

## How It Works

1. **Data Collection**: Utilizes the `twitter_samples` dataset from NLTK, which includes positive and negative tweets.
2. **Data Preprocessing**: Tokenizes tweets and removes usernames.
3. **Feature Extraction**: Incorporates bigram collocations to enhance feature representation.
4. **Model Training**: Trains a Naive Bayes classifier using the processed tweets.
5. **Model Evaluation**: Tests the classifier and calculates its accuracy.

## Technologies Used

- **Python**: The programming language used.
- **NLTK**: Natural Language Toolkit for text processing.
- **Naive Bayes Classifier**: A probabilistic classifier for sentiment analysis.

## Setup

**Install Dependencies**:
   `bash
   pip install nltk

**Download NLTK Data:**
**Open a Python interpreter and run:**

**python**
Copy code
import nltk
nltk.download('twitter_samples')
nltk.download('punkt')
Run the Script:
Save the provided script in a file named twitter_sentiment_analysis.py and execute it:

bash
Copy code
python twitter_sentiment_analysis.py

**Accuracy**
The classifier's accuracy is printed at the end of the script execution.
*99.15


**License**
© 2024 Rajeev Sharma (rajeevsharmamachphy@gmail.com)
