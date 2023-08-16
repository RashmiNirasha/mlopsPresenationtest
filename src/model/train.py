# Import libraries
import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow

# Define functions
def main(args):
    # Enable autologging
    mlflow.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    # Separate features (X) and target (y)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("reg_rate", reg_rate)

        # Train model
        model = LogisticRegression(C=1/reg_rate, solver="liblinear")
        model.fit(X_train, y_train)

        # Log metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)

        # Save the model
        mlflow.sklearn.save_model(model, "model")

def parse_args():
    # Setup arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args

# Run script
if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Initialize MLflow
    mlflow.set_tracking_uri("your_mlflow_tracking_uri")

    # Run main function
    main(args)

    # Add space in logs
    print("*" * 60)
    print("\n\n")
