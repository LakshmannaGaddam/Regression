# src/train.py

import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import json

def train(input_file, model_file):
    # Load data
    df = pd.read_csv(input_file)
    
    # Separate features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, model_file)
    
    # Extract model parameters
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Save model parameters
    model_params = {
        'coefficients': coefficients.tolist(),  # Convert numpy array to list for JSON serialization
        'intercept': intercept.tolist()
    }
    with open('model_params.json', 'w') as f:
        json.dump(model_params, f)
    
    # Generate and print model equation
    feature_names = ['TV', 'Radio', 'Newspaper']
    equation = generate_model_equation(coefficients, intercept, feature_names)
    print("Model Equation:")
    print(equation)

def generate_model_equation(coefficients, intercept, feature_names):
    terms = [f"{coef:.2f} * {name}" for coef, name in zip(coefficients, feature_names)]
    equation = " + ".join(terms)
    equation = f"Sales = {intercept:.2f} + " + equation
    return equation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear regression model")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Model file')
    args = parser.parse_args()

    train(args.input, args.output)
