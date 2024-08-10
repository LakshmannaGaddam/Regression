import argparse
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

def evaluate(input_file, model_file, output_file):
    df = pd.read_csv(input_file)
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    model = joblib.load(model_file)
    #model.params()
    y_pred = model.predict(X)
    
    metrics = {
        'mean_squared_error': mean_squared_error(y, y_pred),
        'r2_score': r2_score(y, y_pred)
    }

    # Save metrics
    with open(output_file, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model', type=str, required=True, help='Model file')
    parser.add_argument('--output', type=str, required=True, help='Metrics file')
    args = parser.parse_args()

    evaluate(args.input, args.model, args.output)
