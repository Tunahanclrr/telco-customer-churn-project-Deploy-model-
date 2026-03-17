import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    # Kullanmayacağımız sütunları düşüyoruz
    cols_to_drop = ['TotalCharges','customerID',"PhoneService"]
    df = df.drop(columns=cols_to_drop)
    print("Columns dropped successfully.")

    # Binary encoding
    df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
    df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print("Binary encoding successfully.")

    # Features ve target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split successfully.")

    # Preprocessor tanımı
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    print("Preprocessor created successfully.")
    return X_train, X_test, y_train, y_test, preprocessor