import pandas as pd
from catboost import CatBoostClassifier

# Constants
MODEL_PATH = "model/catboost_model.cbm"

def load_model(model_path):
    """Load the trained CatBoost model from a file."""
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def predict_churn(model, user_input):
    """Predict the churn probability for a given user input."""
    try:
        # Prepare data for prediction
        user_data = pd.DataFrame([user_input])

        # Ensure all data is encoded in UTF-8
        user_data = user_data.applymap(
            lambda x: str(x).encode("utf-8", "ignore").decode("utf-8")
        )

        # Make prediction
        prediction = model.predict_proba(user_data)[:, 1][0]

        return {"Churn Probability": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

def get_user_input():
    """Retrieve user input for churn prediction."""
    customerID = "6464-UIAEA"
    print("Please enter the following information:")
    gender = input("Gender (Male/Female): ").strip()
    senior_citizen = int(input("Senior Citizen (0/1): ").strip())
    partner = input("Partner (Yes/No): ").strip()
    dependents = input("Dependents (Yes/No): ").strip()
    tenure = int(input("Tenure (months): ").strip())
    phone_service = input("Phone Service (Yes/No): ").strip()
    multiple_lines = input("Multiple Lines (Yes/No): ").strip()
    internet_service = input("Internet Service (DSL/Fiber optic/No): ").strip()
    online_security = input("Online Security (Yes/No): ").strip()
    online_backup = input("Online Backup (Yes/No): ").strip()
    device_protection = input("Device Protection (Yes/No): ").strip()
    tech_support = input("Tech Support (Yes/No): ").strip()
    streaming_tv = input("Streaming TV (Yes/No): ").strip()
    streaming_movies = input("Streaming Movies (Yes/No): ").strip()
    contract = input("Contract (Month-to-month/One year/Two year): ").strip()
    paperless_billing = input("Paperless Billing (Yes/No): ").strip()
    payment_method = input(
        "Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): "
    ).strip()
    monthly_charges = float(input("Monthly Charges ($): ").strip())
    total_charges = float(input("Total Charges ($): ").strip())

    return {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

def main():
    """Main function to execute the churn prediction."""
    model = load_model(MODEL_PATH)
    user_input = get_user_input()
    result = predict_churn(model, user_input)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        formatted_churn_probability = "{:.2%}".format(result["Churn Probability"])
        print(f"Churn Probability: {formatted_churn_probability}")

if __name__ == "__main__":
    main()