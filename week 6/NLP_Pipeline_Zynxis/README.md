# NLP Pipeline for Zynxis

This project builds a simple NLP pipeline for sentiment analysis on student/intern feedback using Python, NLTK, and scikit-learn.

## Project structure

- data/feedback.csv: sample feedback dataset
- src/preprocessing.py: tokenization and stopword removal
- src/vectorization.py: TF-IDF vectorization
- src/model.py: logistic regression classifier
- main.py: pipeline entry point

## How to run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the pipeline:
   python main.py

## What the pipeline demonstrates

- Tokenization
- Stopword removal
- TF-IDF vectorization
- Sentiment classification output
