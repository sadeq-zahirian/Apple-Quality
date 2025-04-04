import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, preprocess_data, standardisation
from src.train_model import train_model, test_model

df = load_data("../datasets/apple_quality.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_list = train_model(X_train, y_train)

for name, model in model_list.items():
    test_model(model, X_test, y_test)



def test_user():
    sample = list(map(float, input("enter data of new apple (by ,) : ").split(",")))
    sample_df = pd.DataFrame([sample])
    sample_scaled = standardisation(sample_df)

    for name, model in model_list.items():
        prediction = model.predict(sample_scaled)[0]
        print(f"model {name}: Good" if prediction == 1 else "Bad")


test_user()
