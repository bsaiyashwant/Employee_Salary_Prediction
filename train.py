from src.preprocess import preprocess_data
import joblib
from sklearn.linear_model import LinearRegression

def train_model():
    X, y, le = preprocess_data('data/salary_data.csv')

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, 'models/salary_model.pkl')
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()