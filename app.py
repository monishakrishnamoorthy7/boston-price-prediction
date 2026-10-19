from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "boston.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURES = [
    {"name": "crim", "label": "Crime rate", "description": "Per-capita crime rate by town", "unit": "rate", "step": "0.001"},
    {"name": "zn", "label": "Residential land", "description": "Share of residential land zoned for large lots", "unit": "%", "step": "0.1"},
    {"name": "indus", "label": "Industrial land", "description": "Share of non-retail business acres", "unit": "%", "step": "0.1"},
    {"name": "chas", "label": "Charles River", "description": "Bounds the river: 1 yes, 0 no", "unit": "0 or 1", "step": "1", "select": True},
    {"name": "nox", "label": "Nitric oxide", "description": "Nitric oxide concentration", "unit": "ppm", "step": "0.001"},
    {"name": "rm", "label": "Rooms per home", "description": "Average number of rooms", "unit": "rooms", "step": "0.01"},
    {"name": "age", "label": "Older homes", "description": "Share of homes built before 1940", "unit": "%", "step": "0.1"},
    {"name": "dis", "label": "Employment distance", "description": "Weighted distance to employment centres", "unit": "index", "step": "0.001"},
    {"name": "rad", "label": "Highway access", "description": "Accessibility to radial highways", "unit": "index", "step": "1"},
    {"name": "tax", "label": "Property tax", "description": "Property-tax rate per $10,000", "unit": "rate", "step": "1"},
    {"name": "ptratio", "label": "Pupil / teacher ratio", "description": "Town-level pupil-teacher ratio", "unit": "ratio", "step": "0.1"},
    {"name": "b", "label": "Historical B index", "description": "Historical dataset feature; interpret with care", "unit": "index", "step": "0.01", "sensitive": True},
    {"name": "lstat", "label": "Lower-status population", "description": "Share of population in the source feature", "unit": "%", "step": "0.1"},
]
DATA = pd.read_csv(DATA_PATH)
for feature in FEATURES:
    feature_range = DATA[feature["name"]].agg(["min", "max"])
    feature["min"] = str(float(feature_range["min"]))
    feature["max"] = str(float(feature_range["max"]))


def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def get_model_insights():
    data = DATA
    feature_names = [feature["name"] for feature in FEATURES]
    x = data[feature_names]
    y = data["medv"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    evaluation_model = LinearRegression().fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    order = np.argsort(y_test.to_numpy())
    chart_rows = [
        {"actual": round(float(y_test.iloc[index]), 2), "predicted": round(float(predictions[index]), 2)}
        for index in order
    ]
    coefficients = [
        {"label": feature["label"], "value": round(float(coefficient), 3)}
        for feature, coefficient in zip(FEATURES, evaluation_model.coef_)
    ]
    return {
        "sample_count": len(data),
        "test_count": len(y_test),
        "metrics": {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
            "r2": round(float(r2_score(y_test, predictions)), 3),
        },
        "chart": chart_rows,
        "coefficients": coefficients,
    }


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "terravista-development-key")
model = load_model()
insights = get_model_insights()


def blank_values():
    return {feature["name"]: "" for feature in FEATURES}


@app.route("/")
def home():
    return render_template(
        "index.html",
        active_page="home",
        title="TerraVista AI — Read the shape of a place.",
        features=FEATURES,
        values=blank_values(),
        prediction=None,
        error=None,
        history=session.get("prediction_history", []),
        insights=insights,
    )


@app.route("/estimate")
def estimate():
    return render_template(
        "index.html",
        active_page="predict",
        title="Estimate a property — TerraVista AI",
        features=FEATURES,
        values=blank_values(),
        prediction=None,
        error=None,
        history=session.get("prediction_history", []),
        insights=insights,
    )


@app.route("/model-notes")
def model_notes():
    return render_template(
        "model_notes.html",
        active_page="model",
        title="Model notes — TerraVista AI",
        insights=insights,
    )


@app.route("/recent")
def recent():
    return render_template(
        "recent.html",
        active_page="recent",
        title="Recent readings — TerraVista AI",
        history=session.get("prediction_history", []),
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = {feature["name"]: request.form.get(feature["name"], "").strip() for feature in FEATURES}
    try:
        data = np.array([float(values[feature["name"]]) for feature in FEATURES], dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError("Values must be finite numbers.")
        for feature, value in zip(FEATURES, data):
            if value < float(feature["min"]) or value > float(feature["max"]):
                raise ValueError(
                    f"{feature['label']} must be between {feature['min']} and {feature['max']}."
                )
        prediction = float(model.predict(pd.DataFrame([data], columns=[feature["name"] for feature in FEATURES]))[0])
    except (TypeError, ValueError):
        return render_template(
            "index.html",
            active_page="predict",
            title="Estimate a property — TerraVista AI",
            features=FEATURES,
            values=values,
            prediction=None,
            error="Please enter a valid number for every field before estimating.",
            history=session.get("prediction_history", []),
            insights=insights,
        ), 400

    prediction_record = {
        "value": round(prediction, 2),
        "rooms": values["rm"],
        "timestamp": pd.Timestamp.now().strftime("%b %d, %H:%M"),
    }
    history = [prediction_record] + session.get("prediction_history", [])
    session["prediction_history"] = history[:5]
    return render_template(
        "estimate.html",
        active_page="predict",
        title="Estimate a property — TerraVista AI",
        features=FEATURES,
        values=values,
        prediction=round(prediction, 2),
        error=None,
        history=session["prediction_history"],
        insights=insights,
    )


@app.post("/clear-history")
def clear_history():
    session.pop("prediction_history", None)
    return redirect(url_for("recent"))


@app.get("/api/insights")
def api_insights():
    return jsonify(insights)


if __name__ == "__main__":
    app.run(debug=True)
