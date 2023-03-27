import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('C:/Users/santi/OneDrive/Escritorio/SD Hackathon/training-stroke.csv')
df = df[df.Gender != 'Other']
df = df.dropna()

X = df[["Age", "BodyMassIndex", "Gender"]]
X = X.replace(["Female", "Male"], [1, 0])

y = df["Stroke"]

clf = LogisticRegression() 
clf.fit(X, y)

import joblib
joblib.dump(clf, "clf.pkl")

import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.header("Machine Learning for Stroke")

# Input bar 1
Age = st.number_input("Enter Age")

# Input bar 2
BodyMassIndex = st.number_input("Enter BodyMassIndex")

# Dropdown input
Gender = st.selectbox("Select Gender", ("Male", "Female"))

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[Age, BodyMassIndex, Gender]], 
                     columns = ["Age", "BodyMassIndex", "Gender"])
    X = X.replace(["Female", "Male"], [1, 0])
    
    # Get prediction
    prediction = clf.predict_proba(X)[:, 1]
    
    # Output prediction
    st.text(f"This instance is a {prediction}")
