import pandas as pd
import joblib

model = joblib.load('xgboost_model.pkl')
features = joblib.load('features.pkl')

def predictions(input_dict):
    df = pd.DataFrame([input_dict])
    output = model.predict(df)
    
    print(f"The prediction of the input is {output[0]}")


inputs = {
    feature: float(input(f"Enter {feature}:"))
    for feature in features
}

predictions(inputs)