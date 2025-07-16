# 💼 Employee Salary Prediction App

This is a Streamlit-based web application that predicts whether an employee earns more than $50K per year using the Adult Income dataset and an XGBoost classifier.

## 🚀 Features

- Predicts employee salary category (`<=50K` or `>50K`) based on personal and professional attributes
- Offers confidence scores for each prediction
- Visualizes dataset distributions and patterns
- Option to retrain the model dynamically from UI
- Uses cached data loading for performance

## 🧠 Model: XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is a powerful and scalable machine learning algorithm, especially effective for structured/tabular data. It's used here for its performance, regularization, and handling of both numeric and categorical data.

## 📊 Dataset

- Source: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- File used: `adult 3.csv`
- Target variable: `salary` (binary: `<=50K` or `>50K`)

## ⚙️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Model**: [XGBoost](https://xgboost.readthedocs.io/)
- **Backend**: Python, Pandas, Seaborn, Matplotlib, Joblib
- **Visualization**: Matplotlib, Seaborn


## 🧪 How It Works

1. Load and preprocess the dataset
2. Encode categorical features
3. Train or load an existing XGBoost model
4. Take user input via Streamlit sidebar
5. Predict salary category and confidence
6. Display visualizations for better understanding

## ▶️ Run the App

Make sure you have Python and the required libraries installed:

```bash
pip install -r requirements.txt
pip install streamlit pandas scikit-learn xgboost matplotlib seaborn joblib
streamlit run app.py
```

## 🔁 Retraining the Model
1. Enable the "Retrain Model" checkbox in the sidebar to:
2. Reprocess data
3. Encode features again
4. Retrain the XGBoost model
5. Save the model and encoders for future use
