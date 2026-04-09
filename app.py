import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("Customer Purchase Predictor")

# Load dataset
df = pd.read_csv("data.csv")

# Features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Train model
model = LogisticRegression()
model.fit(X, y)

# User input
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 60, 25)
salary = st.sidebar.slider("Estimated Salary", 10000, 150000, 50000)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[age, salary]])

    if prediction[0] == 1:
        st.success("Customer is likely to PURCHASE ✅")
    else:
        st.error("Customer is NOT likely to purchase ❌")