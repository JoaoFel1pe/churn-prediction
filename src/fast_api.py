import pandas as pd
import uvicorn
from catboost import CatBoostClassifier
from fastapi import FastAPI

# Constants
MODEL_PATH = "model/catboost_model.cbm"

def load_model():
    """Load the trained CatBoost model from a file."""
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

def get_churn_probability(data, model):
    """Predict churn probability from data in DataFrame format."""
    # Convert incoming data into a DataFrame
    dataframe = pd.DataFrame.from_dict(data, orient="index").T
    # Make the prediction
    churn_probability = model.predict_proba(dataframe)[0][1]
    return churn_probability

# Load the model
model = load_model()

# Create the FastAPI application
app = FastAPI(title="Churn Prediction API", version="1.0")

@app.get("/")
def index():
    """Root endpoint returning a welcome message."""
    return {"message": "CHURN Prediction API"}

@app.post("/predict/")
def predict_churn(data: dict):
    """API endpoint to predict churn probability."""
    # Get the prediction
    churn_probability = get_churn_probability(data, model)
    # Return the prediction
    return {"Churn Probability": churn_probability}

# Run the application
if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="127.0.0.1", port=5000)