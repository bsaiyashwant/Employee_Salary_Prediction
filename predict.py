import joblib
import pandas as pd

def predict_salary(input_data):
    model = joblib.load('models/salary_model.pkl')
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return round(prediction)