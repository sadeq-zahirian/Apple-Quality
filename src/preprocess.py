import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()


def load_data(filepath):
    df = pd.read_csv(filepath)

    return df


def preprocess_data(df):
    df["Acidity"] = pd.to_numeric(df["Acidity"], errors='coerce')
    df.dropna(subset=["Acidity"], inplace=True)

    df["Quality"] = df["Quality"].map({"good": 1, "bad": 0})

    X = df.drop(columns=['Quality', "A_id"])
    y = df['Quality']

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def standardisation(X):
    X = np.array(X).reshape(1, -1)
    return scaler.transform(X)
