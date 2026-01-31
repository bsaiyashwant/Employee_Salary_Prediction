import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path):
    df = pd.read_csv(path)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Education'] = le.fit_transform(df['Education'])
    df['JobTitle'] = le.fit_transform(df['JobTitle'])

    X = df.drop('Salary', axis=1)
    y = df['Salary']

    return X, y, le