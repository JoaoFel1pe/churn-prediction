import joblib
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt

# Constants
MODEL_PATH = "model/catboost_model.cbm"
DATA_PATH = "data/churn_data.parquet"
GENDER_OPTIONS = ("Female", "Male")
YES_NO_OPTIONS = ("No", "Yes")
INTERNET_SERVICE_OPTIONS = ("No", "DSL", "Fiber optic")
CONTRACT_OPTIONS = ("Month-to-month", "One year", "Two year")
PAYMENT_METHOD_OPTIONS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)

# Streamlit page configuration
st.set_page_config(page_title="Churn Project")

@st.cache_resource
def load_data():
    """Load data from a Parquet file."""
    return pd.read_parquet(DATA_PATH)

def load_x_y(file_path):
    """Load and reset index of a joblib file."""
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    """Load the CatBoost model from a file."""
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

def calculate_shap(model, x_train, X_test):
    """Calculate SHAP values for training and test data."""
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(x_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, x_train):
    """Visualize SHAP values for a specific customer."""
    customer_index = X_test[X_test["customerID"] == customer_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6, 6), dpi=200)
    shap.decision_plot(
        explainer.expected_value,
        shap_values_cat_test[customer_index],
        X_test[X_test["customerID"] == customer_id],
        link="logit",
    )
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, x_train):
    """Display a summary plot of SHAP values."""
    shap.summary_plot(shap_values_cat_train, x_train, plot_type="bar", plot_size=(12, 12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    """Display a SHAP waterfall plot."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    """Calculate and display SHAP summary."""
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, data, customer_id, X_train, X_test):
    """Calculate and plot SHAP values for a specific customer."""
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train)
    customer_index = X_test[X_test["customerID"] == customer_id].index[0]
    display_shap_waterfall_plot(
        explainer,
        explainer.expected_value,
        shap_values_cat_test[customer_index],
        feature_names=X_test.columns,
        max_display=20,
    )

def get_user_input(max_tenure, max_monthly_charges, max_total_charges):
    """Retrieve user input for churn prediction."""
    customerID = "6464-UIAEA"
    gender = st.selectbox("Gender:", GENDER_OPTIONS)
    senior_citizen = st.number_input("SeniorCitizen (0: No, 1: Yes)", min_value=0, max_value=1, step=1)
    partner = st.selectbox("Partner:", YES_NO_OPTIONS)
    dependents = st.selectbox("Dependents:", YES_NO_OPTIONS)
    tenure = st.number_input("Tenure:", min_value=0, max_value=max_tenure, step=1)
    phone_service = st.selectbox("PhoneService:", YES_NO_OPTIONS)
    multiple_lines = st.selectbox("MultipleLines:", YES_NO_OPTIONS)
    internet_service = st.selectbox("InternetService:", INTERNET_SERVICE_OPTIONS)
    online_security = st.selectbox("OnlineSecurity:", YES_NO_OPTIONS)
    online_backup = st.selectbox("OnlineBackup:", YES_NO_OPTIONS)
    device_protection = st.selectbox("DeviceProtection:", YES_NO_OPTIONS)
    tech_support = st.selectbox("TechSupport:", YES_NO_OPTIONS)
    streaming_tv = st.selectbox("StreamingTV:", YES_NO_OPTIONS)
    streaming_movies = st.selectbox("StreamingMovies:", YES_NO_OPTIONS)
    contract = st.selectbox("Contract:", CONTRACT_OPTIONS)
    paperless_billing = st.selectbox("PaperlessBilling", YES_NO_OPTIONS)
    payment_method = st.selectbox("PaymentMethod:", PAYMENT_METHOD_OPTIONS)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=max_monthly_charges, step=0.01)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=max_total_charges, step=0.01)
    
    return pd.DataFrame({
        "customerID": [customerID],
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    })

def main():
    """Main function to run the Streamlit app."""
    model = load_model()
    data = load_data()

    X_train = load_x_y("data/X_train.pkl")
    X_test = load_x_y("data/X_test.pkl")
    y_train = load_x_y("data/y_train.pkl")
    y_test = load_x_y("data/y_test.pkl")

    max_tenure = data["tenure"].max()
    max_monthly_charges = data["MonthlyCharges"].max()
    max_total_charges = data["TotalCharges"].max()

    # Radio buttons for options
    election = st.radio(
        "Make Your Choice:",
        ("Feature Importance", "User-based SHAP", "Calculate the probability of CHURN"),
    )
    available_customer_ids = X_test["customerID"].tolist()

    if election == "User-based SHAP":
        customer_id = st.selectbox("Choose the Customer", available_customer_ids)
        customer_index = X_test[X_test["customerID"] == customer_id].index[0]
        st.write(f"Customer {customer_id}: Actual value for the Customer Churn : {y_test.iloc[customer_index]}")
        y_pred = model.predict(X_test)
        st.write(f"Customer {customer_id}: CatBoost Model's prediction for the Customer Churn : {y_pred[customer_index]}")
        plot_shap(model, data, customer_id, X_train=X_train, X_test=X_test)

    elif election == "Feature Importance":
        summary(model, data, X_train=X_train, X_test=X_test)

    elif election == "Calculate the probability of CHURN":
        new_customer_data = get_user_input(max_tenure, max_monthly_charges, max_total_charges)
        confirmation_button = st.button("Confirm")

        if confirmation_button:
            churn_probability = model.predict_proba(new_customer_data)[:, 1]
            formatted_churn_probability = "{:.2%}".format(churn_probability.item())
            big_text = f"<h1>Churn Probability: {formatted_churn_probability}</h1>"
            st.markdown(big_text, unsafe_allow_html=True)
            st.write(new_customer_data.to_dict())

if __name__ == "__main__":
    st.title("Customer Churn Project")
    main()