import numpy as np
import pandas as pd


def create_raw_dataset() -> pd.DataFrame:
    data = {
        "age": [25, 38, None, 44, 52, 27, 19, 80, 33, 120, -5, 45, 60, None, 29],
        "income": [45000, 92000, 120000, None, 78000, 54000, "55k", 150000, 82000, 780000, 35000, 62000, None, 45000, 58000],
        "gender": ["male", "Female", "F", "female", "MALE", None, "m", "FEMALE", "Male", "Male", "female", "unknown", "Female", "male", None],
        "occupation": ["Engineer", "Data Scientist", "engineer", None, "Teacher", "Nurse", "Data_scientist", "teacher", "Engineer", "Lawyer", "Doctor", "doctor", "Nurse", "Engineer", "Teacher"],
        "years_experience": [2, 12, 8, 15, None, 4, 1, 40, 10, 36, 0, 20, 18, 5, None],
        "city": ["New York", "new york", "Chicago", "Los Angeles", "los angeles", None, "Chicago", "Houston", "HOUSTON", "New york", "Boston", "boston", "Houston", "Chicago", "LOS ANGELES"],
        "purchases_last_month": [5, 12, 7, 18, 9, 4, 0, 25, 10, 2, 3, 15, 8, None, 6],
    }
    raw_df = pd.DataFrame(data)
    return raw_df


def clean_gender(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    normalized = normalized.replace({"f": "female", "female": "female", "m": "male", "male": "male", "unknown": np.nan, "none": np.nan})
    return normalized


def clean_occupation(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.lower().str.replace("_", " ")
    normalized = normalized.replace({"engineer": "engineer", "data scientist": "data scientist", "teacher": "teacher", "nurse": "nurse", "lawyer": "lawyer", "doctor": "doctor"})
    return normalized


def clean_city(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.title()
    normalized = normalized.replace({"None": np.nan})
    return normalized


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["years_experience"] = df["years_experience"].fillna(df["years_experience"].median())
    df["income"] = df["income"].astype(float)
    df["income_per_experience"] = df["income"] / (df["years_experience"] + 1)
    df["age_bracket"] = pd.cut(
        df["age"],
        bins=[0, 25, 40, 60, np.inf],
        labels=["Young", "Mid", "Experienced", "Senior"],
        include_lowest=True,
    )
    df["high_value_customer"] = (df["purchases_last_month"] >= 10).astype(int)
    return df


def remove_outliers(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    filter_mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[filter_mask].reset_index(drop=True)


def build_preprocessing_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    before_df = raw_df.copy()

    raw_df["age"] = pd.to_numeric(raw_df["age"], errors="coerce")
    # Validate age: should be between 0 and 120
    raw_df.loc[(raw_df["age"] < 0) | (raw_df["age"] > 120), "age"] = np.nan
    
    raw_df["income"] = raw_df["income"].replace({"55k": 55000}).astype(float)
    raw_df["gender"] = clean_gender(raw_df["gender"])
    raw_df["occupation"] = clean_occupation(raw_df["occupation"])
    raw_df["city"] = clean_city(raw_df["city"])

    raw_df["years_experience"] = pd.to_numeric(raw_df["years_experience"], errors="coerce")
    # Validate years_experience: should not be negative
    raw_df.loc[raw_df["years_experience"] < 0, "years_experience"] = np.nan
    
    raw_df["purchases_last_month"] = pd.to_numeric(raw_df["purchases_last_month"], errors="coerce")

    cleaned_df = raw_df.copy()

    cleaned_df["age"] = cleaned_df["age"].fillna(cleaned_df["age"].median())
    cleaned_df["income"] = cleaned_df["income"].fillna(cleaned_df["income"].median())
    cleaned_df["years_experience"] = cleaned_df["years_experience"].fillna(cleaned_df["years_experience"].median())
    cleaned_df["purchases_last_month"] = cleaned_df["purchases_last_month"].fillna(cleaned_df["purchases_last_month"].median())
    cleaned_df["gender"] = cleaned_df["gender"].fillna("unknown")
    cleaned_df["occupation"] = cleaned_df["occupation"].fillna("unknown")
    cleaned_df = create_engineered_features(cleaned_df)

    numeric_cols = ["age", "income", "years_experience", "purchases_last_month"]
    categorical_cols = ["gender", "occupation", "city"]

    cleaned_df = remove_outliers(cleaned_df, numeric_cols + ["income_per_experience"])

    numeric_medians = cleaned_df[numeric_cols].median()
    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(numeric_medians)

    numeric_scaled = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std()
    one_hot = pd.get_dummies(cleaned_df[categorical_cols].fillna("Unknown"), drop_first=False)

    result_df = pd.concat([numeric_scaled.reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1)
    result_df["income_per_experience"] = cleaned_df["income_per_experience"].values
    result_df["age_bracket"] = cleaned_df["age_bracket"].astype(str).values
    result_df["high_value_customer"] = cleaned_df["high_value_customer"].values

    print("\n----- Raw sample data -----")
    print(before_df.head(10).to_string(index=False))
    print("\n----- Cleaned and engineered sample data -----")
    print(result_df.head(10).to_string(index=False))

    print("\n----- Before summary -----")
    print(before_df.describe(include="all"))
    print("\n----- After summary -----")
    print(result_df.describe(include="all"))

    return result_df


def main():
    raw_df = create_raw_dataset()
    processed_df = build_preprocessing_pipeline(raw_df)
    print(f"\nProcessed dataset has {processed_df.shape[0]} rows and {processed_df.shape[1]} columns.")


if __name__ == "__main__":
    main()
