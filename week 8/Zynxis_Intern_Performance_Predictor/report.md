# Final Capstone Project Report: Zynxis Intern Performance Predictor

## 1. Executive Summary

This capstone project addresses the operational challenge of identifying which interns are likely to perform strongly during their internship period. The project builds an end-to-end machine learning pipeline that collects and transforms internship data, trains a predictive model, evaluates the model, and exposes the result through a user-friendly deployed interface.

The final system predicts one of three performance categories: Excellent, Good, or Needs Improvement. The solution helps Zynxis managers prioritize mentorship, review performance early, and make data-driven workforce decisions.

## 2. Business Problem

Intern performance is currently judged manually using subjective assessments. This creates inconsistency, delays intervention for struggling interns, and makes it harder to identify high performers early. Zynxis needs a practical system that uses measurable signals, such as coding ability, communication, discipline, attendance, and mentor evaluations, to forecast performance more accurately.

## 3. Data Collection and Preparation

The dataset used in this project was created to simulate a realistic internship evaluation scenario. It includes both categorical and numerical variables:

- Domain of the internship
- Education level
- Coding score
- Communication score
- Teamwork score
- Discipline score
- Attendance percentage
- Hours logged
- Project quality
- Prior experience
- Mentor rating
- Performance label

The data is stored in `data/interns.csv` and is loaded using preprocessing logic in `src/preprocessing.py`.

The preparation pipeline includes:

- loading the dataset
- validating required columns
- imputing missing values
- scaling numeric features
- encoding categorical features
- preparing the final feature matrix for the model

## 4. Modeling Approach

A supervised classification model was trained using a `RandomForestClassifier` inside a scikit-learn pipeline.

This model was chosen because it handles mixed data types well, provides strong predictive performance on tabular data, and is interpretable for business stakeholders.

The pipeline structure is:

1. Feature preprocessing
2. Model training
3. Prediction generation

## 5. Training and Evaluation

The model was trained on a structured split of the dataset using a train-test split with stratified sampling to preserve class balance.

Evaluation metrics include:

- Macro F1-score
- Classification report
- Confusion matrix

The trained model is saved in `models/best_model.pkl` and the metadata is saved in `models/model_metadata.json`.

## 6. Deployment Interface

The project includes a `Streamlit` web application in `app.py`. This application provides a simple form where managers can input an intern's profile and receive a predicted performance category with a confidence estimate.

The app offers:

- domain selection
- education input
- technical and behavioral assessments
- predictive result display
- confidence bar chart
- recommendation summary

## 7. Business Value

The implemented system provides several advantages:

- early identification of promising interns
- proactive intervention for weak performers
- improved consistency in performance assessment
- support for better mentoring and task allocation
- a data-driven foundation for workforce planning

## 8. Project Structure

- `data/interns.csv`: dataset
- `models/best_model.pkl`: trained model
- `src/preprocessing.py`: preprocessing code
- `src/train.py`: training pipeline
- `src/evaluate.py`: evaluation logic
- `app.py`: deployed interface
- `README.md`: usage guide
- `notebooks/analysis.ipynb`: exploratory notebook

## 9. Conclusion

This project demonstrates a complete AI/ML lifecycle for a realistic business use case at Zynxis. From dataset preparation to deployment, the solution is designed to be practical, understandable, and immediately usable in a real organizational environment.

The system supports better internship management and creates a strong foundation for future extensions such as internship matching, automated resume screening, or enhanced HR dashboards.
