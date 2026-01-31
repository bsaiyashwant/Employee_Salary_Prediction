from src.train import train_model
from src.predict import predict_salary

train_model()

sample = {
    'Age': 27,
    'Gender': 1,
    'Education': 0,
    'JobTitle': 2,
    'Experience': 3
}

salary = predict_salary(sample)
print(f"Predicted Salary: {salary}")