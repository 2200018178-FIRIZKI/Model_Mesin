from fastapi import FastAPI
from pydantic import BaseModel
from category_classification import train_category_classifier
from anomaly_detection import detect_anomalies
from recommendations import rule_based_recommendation
from data_preprocessing import preprocess_data, clean_text

app = FastAPI()


# Model input untuk prediksi kategori
class TransactionInput(BaseModel):
    title: str
    amount: float


@app.post("/predict/category")
def predict_category(transaction: TransactionInput):
    # Load model & vectorizer (bisa dioptimalkan dengan cache)
    model, tfidf = train_category_classifier("../data/expense_data_1.csv")
    import numpy as np

    # Preprocessing input
    df = preprocess_data("../data/expense_data_1.csv")
    scaler_min = df["amount"].min()
    scaler_max = df["amount"].max()
    amount_scaled = (transaction.amount - scaler_min) / (scaler_max - scaler_min)
    note_clean = clean_text(transaction.title)
    # TF-IDF transform
    X_text = tfidf.transform([note_clean])
    import scipy.sparse

    X_all = scipy.sparse.hstack([X_text, [[amount_scaled]]])
    # Prediksi
    pred = model.predict(X_all)
    return {"predicted_category": pred[0]}


@app.get("/detect/anomaly")
def get_anomalies():
    anomalies = detect_anomalies("../data/expense_data_1.csv")
    # Tampilkan 5 anomali pertama
    return anomalies.head().to_dict(orient="records")


@app.get("/recommendation")
def get_recommendation():
    df = preprocess_data("../data/expense_data_1.csv")
    rekomendasi = rule_based_recommendation(df)
    return {"recommendations": rekomendasi}
