import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import warnings
warnings.filterwarnings("ignore")


def run_credit_scoring(csv_path="../data/bank.csv"):
    print("🚀 Initializing Credit Scoring Research System...\n")

    try:
        # =========================
        # 1. LOAD DATA
        # =========================
        with open(csv_path, "r") as f:
            first_line = f.readline()

        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(csv_path, sep=sep)

        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # =========================
        # 2. TARGET PROCESSING
        # =========================
        if "y" in df.columns:
            df["target"] = df["y"].map({"yes": 1, "no": 0})
        elif "deposit" in df.columns:
            df["target"] = df["deposit"].map({"yes": 1, "no": 0})
        else:
            last_col = df.columns[-1]
            df["target"] = LabelEncoder().fit_transform(df[last_col])

        # =========================
        # 3. FEATURE SELECTION
        # =========================
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        features = [c for c in ["age", "balance", "duration", "campaign"] if c in numeric_cols]

        if not features:
            features = numeric_cols[:4]

        X = df[features].fillna(0)
        y = df["target"]

        print(f"📌 Selected features: {features}")

        # =========================
        # 4. TRAIN TEST SPLIT
        # =========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =========================
        # 5. MODEL TRAINING
        # =========================
        model = GaussianNB()
        model.fit(X_train, y_train)

        print("🧠 Model training completed")

        # =========================
        # 6. EVALUATION
        # =========================
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print("\n📊 MODEL PERFORMANCE")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # =========================
        # 7. SAVE MODEL
        # =========================
        os.makedirs("../models", exist_ok=True)
        import joblib
        joblib.dump(model, "../models/credit_model.pkl")
        print("💾 Model saved to models/credit_model.pkl")

        # =========================
        # 8. HISTOGRAM
        # =========================
        os.makedirs("../results", exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.hist(y_prob, bins=30)
        plt.axvline(0.4, linestyle="--")
        plt.title("Distribution of Default Probabilities")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.savefig("../results/probability_distribution.png")
        plt.close()

        print("📈 Probability distribution saved")

        # =========================
        # 9. ROC CURVE
        # =========================
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig("../results/roc_curve.png")
        plt.close()

        print("📊 ROC curve saved")

        print("\n✅ Research pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Critical error: {e}")


if __name__ == "__main__":
    run_credit_scoring()
