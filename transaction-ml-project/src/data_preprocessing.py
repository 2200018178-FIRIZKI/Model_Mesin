import pandas as pd
import string
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("indonesian")]
    return " ".join(tokens)


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df["Note_clean"] = df["title"].astype(str).apply(clean_text)
    df["amount"] = df["amount"].astype(str).str.replace(",", "").astype(float)
    scaler = MinMaxScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["amount"]])
    df = df.dropna(subset=["category"])
    return df


# Contoh penggunaan
if __name__ == "__main__":
    df = preprocess_data("../data/expense_data_1.csv")
    print(df.head())
