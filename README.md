# Customer Churn Prediction Project

This project uses telecommunications company customer data to predict churn (customer attrition) using **CatBoost** and **Streamlit**. It allows both churn prediction and visualization of feature importance via SHAP values.

## ✨ Features

- 📥 Load and preprocess customer data
- 🏋️‍♂️ Train a classification model using CatBoost
- 📊 Evaluate the model with metrics such as accuracy, recall, AUC-ROC, and more
- 🌐 Interactive interface with Streamlit for churn prediction and feature importance visualization using SHAP

## 🛠️ Technologies Used

- **Python**: Main programming language used in the project 🐍
- **Pandas**: Data manipulation and preprocessing 🐼
- **CatBoost**: Machine learning classifier 🐱
- **SHAP**: Model interpretation through explanatory graphs 📈
- **Streamlit**: Interactive web application 🌟
- **Joblib**: Efficient storage for datasets and models 💾
- **Scikit-Learn**: Data evaluation and splitting 📚
- **Docker**: Containerization for the application 🐳
- **FastAPI**: API for the application 🚀

## 🚀 How to Run the Project

### 📦 Install Dependencies

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 🏋️‍♂️ Train the Model

Run the model training script:

```bash
python src/train_model.py
```

### 🌐 Run the Streamlit Application

Run the Streamlit application:

```bash
streamlit run src/streamlit_app.py
```

You can also run the application using Docker:

```bash
docker-compose up
```

Alternatively, you can run the FastAPI application:

```bash
python src/fast_api.py
```

The application will open in your default web browser.

### Model Explanation with SHAP

The SHAP values provide a visual interpretation of how each feature contributes to the model’s prediction. You can visualize this in the Streamlit app after running the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
