# Model Deployment - Iris Species Web App

This project deploys a simple machine learning web app using Streamlit. The app accepts iris flower measurements and predicts the species.

## Project structure

- app.py: Main Streamlit application
- utils/preprocess.py: Builds the input dataframe
- utils/predict.py: Loads the saved model and returns predictions
- model/: Saved model artifacts
- data/sample_data.csv: Sample iris records

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
