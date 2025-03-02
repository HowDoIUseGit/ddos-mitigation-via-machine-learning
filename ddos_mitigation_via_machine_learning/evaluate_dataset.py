import argparse
import pandas as pd

def calculate_label_percentages(csv_path):

    data = pd.read_csv(csv_path)

    if 'Label' not in data.columns:
        raise ValueError("CSV file must contain a 'Label' column.")

    label_counts = data['Label'].value_counts()

    total_entries = len(data)
    percent_ddos = (label_counts.get('ddos', 0) / total_entries) * 100
    percent_benign = (label_counts.get('Benign', 0) / total_entries) * 100

    print(f"Percentage of 'ddos' entries: {percent_ddos:.2f}%")
    print(f"Percentage of 'Benign' entries: {percent_benign:.2f}%")

    return {
        "percent_ddos": percent_ddos,
        "percent_benign": percent_benign,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the percentage of 'ddos' and 'Benign' entries in a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing a 'Label' column.")
    args = parser.parse_args()

    calculate_label_percentages(args.csv_path)
