import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --------------------------
# Function: Train model
# --------------------------
@st.cache_data   # caching to avoid retraining every time
def train_model():
    df = pd.read_csv("social_network_ads.csv")

    X = df[['age', 'estimated_salary']]
    y = df['purchased']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, df

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Social Network Ads Prediction", layout="centered")
st.title("📊 Social Network Ads Prediction App")
st.write("Predict whether a user will purchase a product")

# Train model & load dataset
model, df = train_model()

# Show dataset preview
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# User input for prediction
st.subheader("🔮 Make a Prediction")
age = st.number_input("Enter Age", min_value=18, max_value=70, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=10000, max_value=200000, value=50000)

if st.button("Predict"):
    prediction = model.predict([[age, salary]])
    probability = model.predict_proba([[age, salary]])[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ User WILL purchase the product (Probability: {probability:.2f}%)")
    else:
        st.error(f"❌ User will NOT purchase the product (Probability: {100 - probability:.2f}%)")

# Optional: Dataset stats
st.subheader("📊 Dataset Statistics")
st.write(df.describe())
