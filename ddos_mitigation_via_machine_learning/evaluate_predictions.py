import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(csv_path):
    data = pd.read_csv(csv_path)

    if 'Label' not in data.columns or 'Predicted Label' not in data.columns:
        raise ValueError("CSV file must contain 'Label' and 'Predicted Label' columns.")

    y_true = data['Label']
    y_pred = data['Predicted Label']

    conf_matrix = confusion_matrix(y_true, y_pred)

    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    true_positives = conf_matrix[1, 1]

    total_samples = len(y_true)
    correct_predictions = true_negatives + true_positives
    incorrect_predictions = false_positives + false_negatives

    percent_correct = (correct_predictions / total_samples) * 100
    percent_incorrect = (incorrect_predictions / total_samples) * 100

    percent_true_negatives = (true_negatives / total_samples) * 100
    percent_true_positives = (true_positives / total_samples) * 100
    percent_false_positives = (false_positives / total_samples) * 100
    percent_false_negatives = (false_negatives / total_samples) * 100

    print(f"Correctly Predicted Benign Traffic (True Negatives): {true_negatives} ({percent_true_negatives:.2f}%)")
    print(f"Correctly Predicted DDoS Attacks (True Positives): {true_positives} ({percent_true_positives:.2f}%)")
    print(f"False Positives (Benign Traffic Predicted as DDoS): {false_positives} ({percent_false_positives:.2f}%)")
    print(f"False Negatives (DDoS Traffic Predicted as Benign): {false_negatives} ({percent_false_negatives:.2f}%)")
    print(f"\nTotal Correct Predictions: {correct_predictions} ({percent_correct:.2f}%)")
    print(f"Total Incorrect Predictions: {incorrect_predictions} ({percent_incorrect:.2f}%)")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {
        "true_negatives": true_negatives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "percent_correct": percent_correct,
        "percent_incorrect": percent_incorrect,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing 'Label' and 'Predicted Label' columns.")
    args = parser.parse_args()

    evaluate_predictions(args.csv_path)