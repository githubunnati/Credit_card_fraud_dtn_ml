import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import joblib
import os

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    return df


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Parse datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]  = df["trans_date_trans_time"].dt.hour
    df["day"]   = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month

    # Age from dob
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # Distance between cardholder and merchant (haversine approx)
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )

    # Encode categorical columns
    le = LabelEncoder()
    for col in ["category", "gender"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # Select features
    features = [
        "amt", "category", "gender", "city_pop",
        "hour", "day", "month", "age", "distance",
        "merch_lat", "merch_long"
    ]
    target = "is_fraud"

    X = df[features]
    y = df[target]
    return X, y, features


# ─────────────────────────────────────────────
# 3.  TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                      n_jobs=-1, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# ─────────────────────────────────────────────
# 4.  EVALUATE
# ─────────────────────────────────────────────
def evaluate(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
            "roc_auc":   round(roc_auc_score(y_test, y_proba)  * 100, 2),
            "report":    classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]),
            "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "model":     model,
        }
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  Accuracy : {results[name]['accuracy']}%")
        print(f"  ROC-AUC  : {results[name]['roc_auc']}%")
        print(results[name]["report"])
    return results


# ─────────────────────────────────────────────
# 5.  SAVE BEST MODEL
# ─────────────────────────────────────────────
def save_best_model(results, scaler, features, save_dir="model"):
    os.makedirs(save_dir, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]
    print(f"\n✅ Best model: {best_name}  (ROC-AUC: {results[best_name]['roc_auc']}%)")

    joblib.dump(best_model, os.path.join(save_dir, "fraud_model.pkl"))
    joblib.dump(scaler,     os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(features,   os.path.join(save_dir, "features.pkl"))
    print(f"   Saved to '{save_dir}/' folder.")
    return best_name


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "fraudTest.csv"   # ← change path if needed

    print("📂 Loading data …")
    df = load_data(DATA_PATH)
    print(f"   Rows: {len(df):,}  |  Fraud %: {df['is_fraud'].mean()*100:.2f}%")

    print("\n⚙️  Preprocessing …")
    X, y, features = preprocess(df)

    # Use a sample for speed (remove/adjust for full training)
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, stratify=y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=features)

    print(f"\n🏋️  Training on {len(X_train):,} samples …")
    models = train_models(X_train, y_train)

    print("\n📊 Evaluating …")
    results = evaluate(models, X_test, y_test)

    save_best_model(results, scaler, features)
    print("\n🎉 Done! Run  streamlit run app.py  to launch the dashboard.")
