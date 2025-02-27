import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from data_preprocessing import preprocess_data

def predict_attacks(new_data_path, model_path='ddos_recognition_model_v3.h5'):
    new_data = pd.read_csv(new_data_path)

    X_scaled, _, label_mapping, scaler, valid_indices = preprocess_data(new_data)
    new_data = new_data.iloc[valid_indices].copy()

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(X_scaled)
    predictions_binary = (predictions > 0.5).astype(int)

    print("Label Mapping:", label_mapping)
    print("Predictions Binary:", predictions_binary)

    predictions_labels = [label_mapping[pred] for pred in predictions_binary.flatten()]

    new_data['Predicted Label'] = predictions_labels

    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DDoS attacks")
    parser.add_argument("new_data_path", type=str, help="Path to the CSV file containing the flow data you want to test.")
    args = parser.parse_args()

    results = predict_attacks(args.new_data_path)

    output_file = 'new_data_with_predictions_v3.csv'
    results.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'")
