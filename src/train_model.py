import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit

# Constants
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PARQUET_PATH = "data/churn_data.parquet"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.cbm")
PKL_DIR = "data"
X_TRAIN_PATH = os.path.join(PKL_DIR, "X_train.pkl")
X_TEST_PATH = os.path.join(PKL_DIR, "X_test.pkl")
Y_TRAIN_PATH = os.path.join(PKL_DIR, "y_train.pkl")
Y_TEST_PATH = os.path.join(PKL_DIR, "y_test.pkl")

def load_and_preprocess_data(data_path):
    """Load and preprocess the data."""
    df = pd.read_csv(data_path)
    
    # Convert TotalCharges to numeric, filling NaN values
    df = df.assign(TotalCharges=pd.to_numeric(df["TotalCharges"], errors="coerce"))
    df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"], inplace=True)
    
    # Convert SeniorCitizen to object
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)
    
    # Replace 'No phone service' and 'No internet service' with 'No' for certain columns
    df["MultipleLines"] = df["MultipleLines"].replace("No internet service", "No")
    columns_to_replace = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for column in columns_to_replace:
        df[column] = df[column].replace("No internet service", "No")
    
    # Convert 'Churn' categorical variable to numeric
    df["Churn"] = df["Churn"].replace({"No": 0, "Yes": 1}).infer_objects(copy=False)
    
    return df

def save_data_as_parquet(df, parquet_path):
    """Save the DataFrame as a Parquet file."""
    df.to_parquet(parquet_path, index=False)

def split_data(df):
    """Split the data into training and testing sets using StratifiedShuffleSplit."""
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
    train_index, test_index = next(strat_split.split(df, df["Churn"]))
    
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
    x_train = strat_train_set.drop("Churn", axis=1)
    y_train = strat_train_set["Churn"].copy()
    
    x_test = strat_test_set.drop("Churn", axis=1)
    y_test = strat_test_set["Churn"].copy()
    
    return x_train, x_test, y_train, y_test

def save_data_as_pkl(x_train, x_test, y_train, y_test):
    """Save the training and testing sets as .pkl files."""
    joblib.dump(x_train, X_TRAIN_PATH)
    joblib.dump(x_test, X_TEST_PATH)
    joblib.dump(y_train, Y_TRAIN_PATH)
    joblib.dump(y_test, Y_TEST_PATH)

def train_catboost_model(x_train, y_train, x_test, y_test, categorical_columns):
    """Train a CatBoost model and return the trained model."""
    cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
    cat_model.fit(x_train, y_train, cat_features=categorical_columns, eval_set=(x_test, y_test))
    return cat_model

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return the evaluation metrics."""
    y_pred = model.predict(x_test)
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "Roc_Auc": round(roc_auc_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
    }
    return metrics

def save_model(model, model_path):
    """Save the trained model to a file."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_model(model_path)

def main():
    # Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Save the regulated churn data to a Parquet file
    save_data_as_parquet(df, PARQUET_PATH)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = split_data(df)
    
    # Save the training and testing sets as .pkl files
    save_data_as_pkl(x_train, x_test, y_train, y_test)
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns.to_list()
    
    # Train CatBoost model
    cat_model = train_catboost_model(x_train, y_train, x_test, y_test, categorical_columns)
    
    # Evaluate the model
    metrics = evaluate_model(cat_model, x_test, y_test)
    
    # Print evaluation metrics
    result = pd.DataFrame(metrics, index=["CatBoost_Model"])
    print(result)
    
    # Save the trained model
    save_model(cat_model, MODEL_PATH)

if __name__ == "__main__":
    main()