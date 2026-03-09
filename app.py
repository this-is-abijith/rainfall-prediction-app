from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "model.pkl"),     "rb") as f: model  = pickle.load(f)
with open(os.path.join(BASE, "scaler.pkl"),    "rb") as f: scaler = pickle.load(f)
with open(os.path.join(BASE, "features.json"), "r")  as f: meta   = json.load(f)

FEATURES    = meta["features"]
IMPORTANCES = meta["importances"]
MODEL_ACC   = meta["accuracy"]

@app.route("/")
def index():
    return render_template("index.html", accuracy=MODEL_ACC)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        humidity     = float(data["humidity"])
        temperature  = float(data["temperature"])
        pressure     = float(data["pressure"])
        wind_speed   = float(data["wind_speed"])
        cloud_cover  = float(data["cloud_cover"])

        dew_point     = temperature - ((100 - humidity) / 5)
        temp_humidity = temperature * humidity / 100

        input_vec    = np.array([[humidity, temperature, pressure,
                                  wind_speed, cloud_cover,
                                  dew_point, temp_humidity]])
        input_scaled = scaler.transform(input_vec)
        prob_arr     = model.predict_proba(input_scaled)[0]
        rain_prob    = round(float(prob_arr[1]) * 100, 1)
        prediction   = "Rain" if rain_prob >= 50 else "No Rain"

        if rain_prob >= 75:
            desc = "High chance of heavy rainfall. Carry an umbrella!"
        elif rain_prob >= 50:
            desc = "Moderate chance of rain. Stay prepared."
        elif rain_prob >= 25:
            desc = "Low chance of rain. Mostly dry conditions expected."
        else:
            desc = "Very unlikely to rain. Clear skies ahead!"

        return jsonify({
            "prediction":    prediction,
            "probability":   rain_prob,
            "description":   desc,
            "dew_point":     round(dew_point, 1),
            "temp_humidity": round(temp_humidity, 1),
            "importances":   IMPORTANCES,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/model-info")
def model_info():
    return jsonify({
        "algorithm":   "Random Forest + Gradient Boosting (Soft Voting Ensemble)",
        "features":    FEATURES,
        "accuracy":    MODEL_ACC,
        "importances": IMPORTANCES,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)