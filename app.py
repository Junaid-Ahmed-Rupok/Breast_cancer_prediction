import pandas as pd
import streamlit as st
import joblib

st.title('Breast Cancer Prediction')
model = joblib.load('xgboost_model.pkl')
features = joblib.load('features.pkl')

def predictions_(data):
    df = pd.DataFrame([data])
    output = model.predict(df)
    
    return output[0]

inputs = {
    feature: st.number_input(f"Enter {feature}")
    for feature in features
}




if st.button("Prediction"):
    x = predictions_(inputs)
    st.success(f"The prediction of the given input is {x}")
    

