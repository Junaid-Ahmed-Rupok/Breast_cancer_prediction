import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib

def model_construction(read_csv):
    df = pd.read_csv(read_csv)
    
    # dropping the columns
    df = df.drop(columns=['Unnamed: 32'])
    
    # dealing with the null values
    if df.isna().sum().sum() > 0:
        df = df.apply(
            lambda cols: cols.fillna(cols.mean())
            if np.issubdtype(cols.dtype, np.number)
            else cols.fillna(cols.mode()[0])
        )
    
    # dealing with the duplicates
    df = df.drop_duplicates()
    
    # dealing with the categorical values
    cat_values = df.select_dtypes(include=['object']).columns.tolist()
    encode = OrdinalEncoder()
    df[cat_values] = encode.fit_transform(df[cat_values])
    
    # splitting the data
    x = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # feature extraction
    feature_model = XGBClassifier(random_state=42)
    feature_model.fit(x_train, y_train)
    
    xgboost_importance = feature_model.feature_importances_
    xgboost_series = pd.Series(xgboost_importance, index=x_train.columns).sort_values(ascending=False)
    features = xgboost_series.head(10).index.tolist()
    
    # training the model
    xgboost_model = XGBClassifier(n_estimators=200, learning_rate=0.2, random_state=42)
    xgboost_model.fit(x_train[features], y_train)
    predictions = xgboost_model.predict(x_test[features])
    
    # metrics
    a_score = accuracy_score(y_test, predictions)
    p_score = precision_score(y_test, predictions)
    r_score = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)
    
    print(f"Accuracy Score = {a_score}")
    print(f"Precision Score = {p_score}")
    print(f"Recall = {r_score}")
    print(f"f1 Score = {f_score}")
    
    # saving the model and features  
    joblib.dump(xgboost_model, 'xgboost_model.pkl')
    joblib.dump(features, 'features.pkl')


model_construction(r"C:\Users\JUNAID AHMED\Downloads\datasets\breast_cancer.csv")