import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

def train(input_file, model_file):
    df = pd.read_csv(input_file)
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
    y_pred = model.predict(X_test)
    metrics = {
        'mean_squared_error': mean_squared_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }

    # Save model
    joblib.dump(model, model_file)
    
    # Save metrics
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Model file')
    args = parser.parse_args()

    train(args.input, args.output)

