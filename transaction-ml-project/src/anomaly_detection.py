import pandas as pd
from sklearn.ensemble import IsolationForest
from data_preprocessing import preprocess_data


def extract_time_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["hour"] = df["Date"].dt.hour
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"] >= 5
    return df


def detect_anomalies(data_path):
    df = preprocess_data(data_path)
    df = extract_time_features(df)
    features = ["Amount_scaled", "hour", "dayofweek"]
    X = df[features]

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso_forest.fit_predict(X)
    anomalies = df[df["anomaly"] == -1]
    print("Anomali terdeteksi:")
    print(anomalies[["Date", "title", "amount", "category"]])
    return anomalies


if __name__ == "__main__":
    anomalies = detect_anomalies("../data/expense_data_1.csv")
