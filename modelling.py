import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # [GAP PLUG] Robust path handling
    # This ensures the script finds the CSV whether running locally or in CI
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    file_path = sys.argv[3] if len(sys.argv) > 3 else default_path

    print(f"Loading data from: {file_path}") 
    data = pd.read_csv(file_path)
    
    # Basic Preprocessing
    X = data.drop("Credit_Score", axis=1)
    y = data["Credit_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Capture Hyperparameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 35

    with mlflow.start_run():
        # Train
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Log
        mlflow.sklearn.log_model(model, "model")
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", n_estimators)
        
        print(f"Model trained with accuracy: {accuracy}")