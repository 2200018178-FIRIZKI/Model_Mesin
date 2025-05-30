import pandas as pd
from data_preprocessing import preprocess_data


def aggregate_monthly_expense(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby(["month", "category"])["amount"].sum().reset_index()
    return monthly


def rule_based_recommendation(df):
    monthly = aggregate_monthly_expense(df)
    saran = []
    for _, row in monthly.iterrows():
        if row["category"].lower() == "kopi" and row["amount"] > 200_000:
            saran.append(
                f"Pengeluaran kopi bulan {row['month']} sebesar Rp{int(row['amount'])}. Coba kurangi konsumsi kopi!"
            )
        if row["category"].lower() == "food" and row["amount"] > 1_000_000:
            saran.append(
                f"Pengeluaran makanan bulan {row['month']} sudah Rp{int(row['amount'])}. Pertimbangkan masak sendiri!"
            )
    return saran


if __name__ == "__main__":
    df = preprocess_data("../data/expense_data_1.csv")
    rekomendasi = rule_based_recommendation(df)
    for saran in rekomendasi:
        print(saran)
