import pandas as pd

def load_data(filepath):
    """Load the housing dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Clean and preprocess the housing data."""
    # Convert yes/no columns to 1/0
    yes_no_cols = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'prefarea'
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Convert 'furnishingstatus' to numeric using one-hot encoding
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # Drop rows with missing values (if any)
    df = df.dropna()

    return df

if __name__ == "__main__":
    data_path = "data/Housing.csv"  # Adjust path if needed
    df = load_data(data_path)
    df = preprocess_data(df)
    print(df.head())