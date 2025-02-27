import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data):
    print("================== Starting Data Preprocessing ==================")

    print("Dropping empty columns...")
    data = data.dropna()

    print("Dropping irrelevant columns...")
    columns_to_drop = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Len', 'Bwd Header Len', 'Fwd Seg Size Min', 'Fwd Seg Size Avg',
        'Bwd Seg Size Avg', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
        'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
        'Idle Max', 'Idle Min'
    ]
    data = data.drop(columns=columns_to_drop)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    print("Mapping labels...")
    label_mapping = {v: k for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    print("Label Mapping:", label_mapping)

    print("Encoding categorical features...")
    data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

    print("Splitting features and target...")
    X = data.drop(columns=['Label'])
    y = data['Label']

    print("Normalizing numerical features...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y[X.index]

    print("Applying scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("================== Finished Data Preprocessing ==================")

    return X_scaled, y, label_mapping, scaler, X.index
