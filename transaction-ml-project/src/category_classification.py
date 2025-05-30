import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from data_preprocessing import preprocess_data


def train_category_classifier(data_path):
    df = preprocess_data(data_path)
    X_text = df["Note_clean"]
    X_amount = df[["Amount_scaled"]]
    y = df["category"]

    tfidf = TfidfVectorizer()
    X_text_tfidf = tfidf.fit_transform(X_text)

    import scipy.sparse

    X_all = scipy.sparse.hstack([X_text_tfidf, X_amount])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, tfidf


if __name__ == "__main__":
    model, tfidf = train_category_classifier("../data/expense_data_1.csv")
