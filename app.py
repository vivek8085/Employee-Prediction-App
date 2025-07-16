import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

page = st.sidebar.selectbox("Menu", ["🔍 Prediction", "📘 Algorithm Info"])

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"
COLUMNS_PATH = "model_columns.pkl"
CSV_PATH = "adult 3.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    if "income" in df.columns:
        df.rename(columns={"income": "salary"}, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

if page == "🔍 Prediction":
    st.title("Employee Salary Prediction App Using XGBoost")

    retrain = st.sidebar.checkbox("🔁 Retrain Model")

    if retrain or not os.path.exists(MODEL_PATH):
        st.info("Training model...")

        df_train = df.copy()
        df_train['salary'] = df_train['salary'].str.strip().str.lower().map({'<=50k': 0, '>50k': 1})
        df_train.dropna(subset=['salary'], inplace=True)

        categorical_cols = df_train.select_dtypes(include='object').columns
        encoder = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col])
            encoder[col] = list(le.classes_)

        joblib.dump(encoder, ENCODER_PATH)

        X = df_train.drop("salary", axis=1)
        y = df_train["salary"]
        columns = X.columns.tolist()
        joblib.dump(columns, COLUMNS_PATH)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, MODEL_PATH)

        st.success(f"Model trained with accuracy: {acc:.2%}")
    else:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        columns = joblib.load(COLUMNS_PATH)

    st.sidebar.header("📝 Enter Employee Data")
    def user_input():
        input_data = {
            "age": st.sidebar.slider("Age", 17, 90, 35),
            "workclass": st.sidebar.selectbox("Workclass", encoder["workclass"]),
            "education": st.sidebar.selectbox("Education", encoder["education"]),
            "marital-status": st.sidebar.selectbox("Marital Status", encoder["marital-status"]),
            "occupation": st.sidebar.selectbox("Occupation", encoder["occupation"]),
            "relationship": st.sidebar.selectbox("Relationship", encoder["relationship"]),
            "race": st.sidebar.selectbox("Race", encoder["race"]),
            "gender": st.sidebar.selectbox("Gender", encoder["gender"]),
            "native-country": st.sidebar.selectbox("Native Country", encoder["native-country"]),
            "hours-per-week": st.sidebar.slider("Hours per Week", 1, 99, 40),
            "capital-gain": st.sidebar.number_input("Capital Gain", 0),
            "capital-loss": st.sidebar.number_input("Capital Loss", 0)
        }
        return pd.DataFrame([input_data])

    input_df = user_input()
    for col in input_df.columns:
        if col in encoder:
            input_df[col] = input_df[col].apply(lambda x: encoder[col].index(x))

    input_encoded = input_df.reindex(columns=columns, fill_value=0)

    st.subheader("📊 Salary Prediction")
    if st.button("🔍 Predict Salary"):
        prediction = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0]
        label = ">50K" if prediction == 1 else "<=50K"
        st.success(f"Predicted Salary: {label}")
        st.info(f"Confidence: {prob[1]*100:.2f}% >50K | {prob[0]*100:.2f}% <=50K")

    st.subheader("📈 Dataset Visualizations")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="salary", ax=ax1)
    ax1.set_title("Salary Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="age", y="hours-per-week", hue="salary", ax=ax2)
    ax2.set_title("Age vs Hours-per-week by Salary")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x="education", hue="salary", ax=ax3)
    ax3.set_title("Salary by Education")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

elif page == "📘 Algorithm Info":
    st.title("📘 Algorithm: XGBoost Classifier")

    st.markdown("""
### What is XGBoost?
XGBoost (Extreme Gradient Boosting) is a fast, regularized, and scalable tree-based algorithm. It's widely used in structured/tabular datasets for classification and regression tasks.

---

### Why XGBoost is Used Here?

- Handles both categorical and numerical data well  
- Automatically handles missing values  
- Includes regularization to prevent overfitting  
- Fast and scalable (parallel processing)

---

### Workflow / Pseudocode

Input:
    - Training data: D = {(x₁, y₁), ..., (xₙ, yₙ)}
    - Number of boosting rounds: T
    - Learning rate: η
    - Loss function: L(y, ŷ)

Initialize:
    - F₀(x) = constant value (e.g., log(odds of positive class))

For t = 1 to T:

    1. For each sample i in D:
         - Compute gradient: gᵢ = ∂L(yi, Fₜ₋₁(xᵢ)) / ∂Fₜ₋₁(xᵢ)
         - Compute hessian: hᵢ = ∂²L(yi, Fₜ₋₁(xᵢ)) / ∂Fₜ₋₁(xᵢ)²

    2. Fit a regression tree hₜ(x) to the data {(xᵢ, gᵢ, hᵢ)}:
         - Use gain-based criteria to split nodes (maximize reduction in loss)
         - Use regularization on leaf weights to avoid overfitting

    3. Compute optimal leaf weights wⱼ for each leaf j:
         wⱼ = -Σgᵢ / (Σhᵢ + λ) for samples i in leaf j

    4. Update model:
         Fₜ(x) = Fₜ₋₁(x) + η * hₜ(x)

Output:
    Final model F_T(x)

---

### ✅ Advantages of XGBoost

- High accuracy and performance
- Built-in cross-validation support
- Works well on noisy data
- Avoids overfitting via regularization (L1/L2)

---

### 📉 Where It's Not Ideal

- Slower than simple models like Logistic Regression
- Requires tuning for best results
- Can overfit if used without care

---

### 📘 Resources to Learn More

- [XGBoost Official Docs](https://xgboost.readthedocs.io/)
- [XGBoost vs Random Forest](https://www.analyticsvidhya.com/blog/2020/03/battle-of-the-boosting-algorithms-xgboost-vs-lightgbm/)
- [XGBoost Python Guide](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/)
""")