# 🌧️ RainSense — Rainfall Prediction App

A machine learning web app that predicts whether it will rain based on weather parameters. Built with Flask, scikit-learn, and a clean dark UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- 🔮 Predicts rainfall probability from 5 weather inputs
- 🤖 Ensemble ML model (Random Forest + Gradient Boosting)
- 📊 Live feature importance chart after each prediction
- 🌧️ Animated rain UI with probability arc gauge
- ⚡ REST API endpoint for predictions
- 🧪 Auto-derives dew point and temp-humidity index

---

## 🧠 ML Model

| Property | Detail |
|---|---|
| Algorithm | Soft Voting Ensemble (RF + GBM) |
| Features | Humidity, Temperature, Pressure, Wind Speed, Cloud Cover + 2 derived |
| Training samples | 5,000 |
| Test accuracy | ~83% |
| Output | Rain / No Rain + probability % |

**Why this ensemble?**
- Random Forest handles noisy inputs and avoids overfitting
- Gradient Boosting corrects the RF's weak predictions iteratively
- Soft voting averages probabilities for a more confident final result

---

## 📁 Project Structure

```
rainfall_app/
├── app.py              # Flask backend — routes & prediction logic
├── train_model.py      # Train & save the ML model
├── model.pkl           # Trained ensemble model
├── scaler.pkl          # StandardScaler for feature normalization
├── features.json       # Feature names, importances, accuracy
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend UI
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rainfall-prediction-app.git
cd rainfall-prediction-app
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```
This generates `model.pkl`, `scaler.pkl`, and `features.json` in the project root.

### 5. Run the app
```bash
python app.py
```

Open your browser at **http://localhost:5000** 🎉

---

## 🔌 API

### `POST /predict`

**Request body:**
```json
{
  "humidity": 80,
  "temperature": 22,
  "pressure": 1005,
  "wind_speed": 20,
  "cloud_cover": 75
}
```

**Response:**
```json
{
  "prediction": "Rain",
  "probability": 72.4,
  "description": "High chance of heavy rainfall. Carry an umbrella!",
  "dew_point": 18.4,
  "temp_humidity": 17.6,
  "importances": { "humidity": 0.304, "cloud_cover": 0.151 }
}
```

### `GET /model-info`
Returns model metadata — algorithm, features, accuracy, and importances.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (RandomForest + GradientBoosting VotingClassifier)
- **Data:** NumPy, Pandas
- **Frontend:** Vanilla HTML / CSS / JS

---

## 📈 Future Improvements

- [ ] Connect to a real weather API (OpenWeatherMap)
- [ ] Add a prediction history log
- [ ] Deploy to Render / Railway / Hugging Face Spaces
- [ ] Try LSTM for time-series based prediction

---

## 👤 Author

**Abijith Binu**  
GitHub:https://github.com/this-is-abijith

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).