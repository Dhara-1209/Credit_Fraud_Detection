import streamlit as st
import numpy as np
import pickle

# Load model files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
indices = pickle.load(open("features.pkl", "rb"))


# -------------------------------
# SAMPLE INPUT BUTTON
# -------------------------------
st.subheader("🧪 Try Sample Input")

sample_data = "-1.359807,-0.072781,2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.551600,-0.617801,-0.991390,-0.311169,1.468177,-0.470401,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,-0.110474,0.066928,0.128539,-0.189115,0.133558,-0.021053,149.62"

if st.button("Use Sample Data"):
    st.session_state["input_data"] = sample_data

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("✍️ Enter Transaction Details")

st.warning("⚠️ Enter 29 comma-separated values (V1–V28 + Amount)")

input_data = st.text_area(
    "Input:",
    value=st.session_state.get("input_data", "")
)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    try:
        values = np.array([float(i) for i in input_data.split(",")])

        if len(values) != 29:
            st.warning("⚠️ Please enter exactly 29 values!")
        else:
            values = values.reshape(1, -1)

            # Scale
            values = scaler.transform(values)

            # Feature selection
            values = values[:, indices]

            # Predict
            prediction = model.predict(values)

            st.subheader("🔍 Prediction Result")

            if prediction[0] == 1:
                st.error("🚨 Fraud Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

    except:
        st.error("❌ Invalid input! Please enter numeric values only.")

