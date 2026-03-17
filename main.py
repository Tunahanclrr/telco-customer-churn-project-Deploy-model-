# main.py
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
import joblib
import os

def main():
    df = load_data()
    print("First 5 rows of the dataset:")
    print(df.head())

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    pipe = train_model(X_train, y_train, preprocessor)

    results = evaluate_model(pipe, X_test, y_test)

    print("\nEvaluation Results:")
    print('Confusion Matrix:\n', results['confusion_matrix'])
    print('Classification Report:\n', results['classification_report'])
    print(f"ROC AUC Score: {results['roc_auc_score']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Probabilities for the first 5 test samples:")
    print('proba ilk 5 test örneği için olasılıklar:\n', results['probabilities'][:5])

    ## modeli kaydetme
    model_path = "models/lgbm_churn_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()