"""
train_model.py
==============
Trains a Random Forest + XGBoost ensemble on synthetic weather data.
Run this once before starting the Flask app: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

# ── 1. Generate realistic synthetic weather dataset ───────────────────────────
np.random.seed(42)
N = 5000

def generate_weather_data(n):
    humidity        = np.random.uniform(20, 100, n)
    temperature     = np.random.uniform(5, 45, n)
    pressure        = np.random.uniform(980, 1025, n)
    wind_speed      = np.random.uniform(0, 60, n)
    cloud_cover     = np.random.uniform(0, 100, n)
    dew_point       = temperature - ((100 - humidity) / 5)
    temp_humidity   = temperature * humidity / 100

    # Rainfall probability formula (domain logic baked in)
    rain_prob = (
        0.35 * (humidity / 100) +
        0.20 * (cloud_cover / 100) +
        0.15 * (1 - (pressure - 980) / 45) +
        0.10 * (wind_speed / 60) +
        0.10 * ((dew_point + 10) / 55) +
        0.10 * (temp_humidity / 45)
    )
    rain_prob = np.clip(rain_prob + np.random.normal(0, 0.08, n), 0, 1)
    rainfall = (rain_prob > 0.5).astype(int)

    return pd.DataFrame({
        "humidity":      humidity,
        "temperature":   temperature,
        "pressure":      pressure,
        "wind_speed":    wind_speed,
        "cloud_cover":   cloud_cover,
        "dew_point":     dew_point,
        "temp_humidity": temp_humidity,
        "rainfall":      rainfall,
    })

df = generate_weather_data(N)
print(f"Dataset: {N} samples | Rain days: {df['rainfall'].sum()} ({df['rainfall'].mean()*100:.1f}%)")

# ── 2. Features & target ──────────────────────────────────────────────────────
FEATURES = ["humidity", "temperature", "pressure", "wind_speed",
            "cloud_cover", "dew_point", "temp_humidity"]
X = df[FEATURES]
y = df["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Build Voting Ensemble (RF + GBM) ───────────────────────────────────────
rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gbm", gbm)],
    voting="soft",          # average predicted probabilities
    weights=[1, 1]
)

print("Training ensemble model …")
ensemble.fit(X_train_s, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = ensemble.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

# ── 6. Feature importances (from RF component) ────────────────────────────────
rf_trained = ensemble.estimators_[0]
importances = dict(zip(FEATURES, rf_trained.feature_importances_.tolist()))
importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
print("\nFeature Importances:", importances)

# ── 7. Save artefacts ─────────────────────────────────────────────────────────
with open("model.pkl",   "wb") as f: pickle.dump(ensemble, f)
with open("scaler.pkl",  "wb") as f: pickle.dump(scaler, f)
with open("features.json", "w") as f:
    json.dump({"features": FEATURES, "importances": importances, "accuracy": round(acc*100, 2)}, f)

print("\n✅  Saved: model.pkl | scaler.pkl | features.json")