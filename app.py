import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# 1. Load & Train Model
# ----------------------------
@st.cache_resource
def load_model_and_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, housing.feature_names, X_test, y_test

model, feature_names, X_test, y_test = load_model_and_data()

# ----------------------------
# 2. Streamlit UI
# ----------------------------
st.set_page_config(page_title="California Housing Price Prediction", layout="wide")
st.title("🏡 California Housing Price Prediction App")

st.write("### Enter housing details to predict median house value")

inputs = []
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        if feature == "Longitude":
            value = st.number_input(
                f"{feature}", min_value=-125.0, max_value=-113.0, value=-122.0, step=0.01
            )
        elif feature == "Latitude":
            value = st.number_input(
                f"{feature}", min_value=32.0, max_value=43.0, value=37.0, step=0.01
            )
        else:
            value = st.number_input(f"{feature}", min_value=0.0, value=1.0, step=0.1)
        inputs.append(value)

if st.button("🔮 Predict Price"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"💰 Predicted Median House Value: **${prediction*100000:.2f}**")

# ----------------------------

