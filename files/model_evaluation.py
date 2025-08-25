import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model():
    model = joblib.load("models/house_price_model.pkl")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

if __name__ == "__main__":
    evaluate_model()