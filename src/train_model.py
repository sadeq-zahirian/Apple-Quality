import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report




def train_model(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(max_depth= 10, max_features= 'sqrt', min_samples_split= 5, n_estimators= 200,random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
        "Logistic Regression": LogisticRegression(max_iter=200)
    }

    model_list = {}
    for name, model in models.items():
        if not os.path.exists(f"saved_models/{name}.pkl"):
            model.fit(X_train, y_train)
            model_list[name] = model
            joblib.dump(model, f"saved_models/{name}.pkl")
            print(f"training model {name} ...")
        else:
            model_list[name] = joblib.load(f"saved_models/{name}.pkl")
            print(f"loaded model {name}")

    return model_list


def test_model(model,X_test,y_test):
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

