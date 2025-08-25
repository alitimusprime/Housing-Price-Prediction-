import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
from data_preprocessing import load_data, preprocess_data

def train_model():
    data_path = "data/Housing.csv"
    df = load_data(data_path)
    df = preprocess_data(df)

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save the model and test data for evaluation
    joblib.dump(model, "models/house_price_model.pkl")
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()