import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🏥 Disease Prediction App")

# Load dataset
df = pd.read_csv("hospital_data.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Show group statistics
st.subheader("Average Values by Disease")
st.write(df.groupby("Disease").mean())

# Features and target
X = df[["Age", "Fever", "BP", "Sugar"]]
y = df["Disease"]

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
st.subheader("Model Accuracy")
st.write(f"Accuracy of Model: {accuracy * 100:.2f}%")

# User input for prediction
st.subheader("🔍 Predict Disease")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
fever = st.number_input("Fever (0 = No, 1 = Yes)", min_value=0, max_value=1, value=1)
bp = st.number_input("Blood Pressure Level", min_value=0, max_value=10, value=3)
sugar = st.number_input("Sugar Level", min_value=0, max_value=10, value=4)

if st.button("Predict"):
    new_data = [[age, fever, bp, sugar]]
    prediction = model.predict(new_data)
    if prediction[0] == 1:
        st.success("Disease : NO")
    else:
        st.error("Disease : YES")
