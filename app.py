import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("domain_dataset.csv")

X = df[["Coding", "Math", "Creativity", "Communication"]]
y = df["Domain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.title("🎯 Domain Recommendation System")
st.write("Model Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
st.subheader("Enter your skills to get domain recommendation")
coding = st.slider("Coding", 1, 10)
math = st.slider("Math", 1, 10)
creativity = st.slider("Creativity", 1, 10)
communication = st.slider("Communication", 1, 10)

if st.button("Predict"):
    new_data = pd.DataFrame([[coding, math, creativity, communication]],
                            columns=["Coding", "Math", "Creativity", "Communication"])
    
    result = model.predict(new_data)
    
    st.success(f"Recommended Domain: {result[0]}")