# Zynxis Intern Performance Predictor

This project builds an end-to-end machine learning solution to predict the likely performance category of an intern at Zynxis based on their technical, behavioral, and training data.

## Problem Statement

Zynxis needs a fast and objective way to estimate whether an intern is likely to perform at an excellent, good, or needs-improvement level. A predictive model helps managers identify strong performers early and allocate mentorship effectively.

## Project Goals

- Collect a realistic internship dataset
- Prepare and transform data for modeling
- Train a robust classification model
- Evaluate performance and explain results
- Deploy an interactive prediction interface

## Data

The dataset is stored in `data/interns.csv` and includes the following features:

- domain
- education_level
- coding_score
- communication_score
- teamwork_score
- discipline_score
- attendance_pct
- hours_logged
- project_quality
- prior_experience
- mentor_rating
- performance_label

## Model

The project trains a `RandomForestClassifier` using a preprocessing pipeline with:

- median imputation for numeric features
- most-frequent imputation for categorical features
- standard scaling for numeric features
- one-hot encoding for categorical features

## Run Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train.py
   ```

3. Evaluate the model:
   ```bash
   python src/evaluate.py
   ```

4. Run the local web app:
   ```bash
   streamlit run app.py
   ```

## Folder Structure

- `data/` - internship dataset
- `models/` - serialized trained model
- `src/` - preprocessing, training, and evaluation logic
- `notebooks/` - analysis notebook
- `app.py` - Streamlit deployment interface

## Usage

Open the UI, enter intern metrics, and the app predicts the likely performance label with an associated confidence score.

## Business Impact

This project supports data-driven workforce planning, early intervention for underperforming interns, and better manager allocation for coaching and assignment decisions.
