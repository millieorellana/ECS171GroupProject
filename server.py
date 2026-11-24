"""Live demo: a Flask server that runs the saved models on a person sent from an HTML page

The model notebooks save their trained models to models/ (save_model) and their
test-set predictions to results/ (save_predictions). This server loads both.

Run with: python server.py, then open http://127.0.0.1:5000
"""

import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from preprocessing import FEATURES, MODELS_DIR, RESULTS_DIR, load_adult, onehot, to_categories

# Name used by save_model/save_predictions -> (display name, preprocessing the model was trained on)
MODELS = {
    "naiveBayes": ("Naive Bayes", to_categories),
    "logisticRegression": ("Logistic Regression", onehot),
    "randomForest": ("Random Forest", onehot),
}

trainDf, testDf = load_adult()
yTest = testDf["income"].to_numpy()
NUMERIC = [f for f in FEATURES if pd.api.types.is_numeric_dtype(trainDf[f])]
CATEGORIES = {f: sorted(trainDf[f].unique()) for f in FEATURES if f not in NUMERIC}


def loadModels():
    """{model: {noise %: trained model}} from every models/<model>_noise<NN>.joblib"""
    models = {model: {} for model in MODELS}
    for path in MODELS_DIR.glob("*_noise*.joblib"):
        model, percent = re.fullmatch(r"(\w+)_noise(\d+)", path.stem).groups()
        if model in models:
            models[model][int(percent)] = joblib.load(path)
    return models


def loadErrors():
    """{model: {noise %: test error of each saved run}} from results/<model>.npz"""
    errors = {}
    for model in MODELS:
        path = RESULTS_DIR / f"{model}.npz"
        if path.exists():
            with np.load(path) as data:
                percents = np.round(data["noise"] * 100).astype(int)
                wrong = (data["probs"] > 0.5) != yTest  # predict() picks >50K when P(>50K) > 0.5
            errors[model] = {p: wrong[percents == p].mean(axis=1) for p in sorted(set(percents))}
    return errors


models = loadModels()  # loaded once: restart the server after re-running a notebook
noisePercents = sorted({p for byNoise in models.values() for p in byNoise}) or [0]
testInputs = {model: transform(trainDf, testDf)[1] for model, (_, transform) in MODELS.items()}
testProbs = {}  # (model, noise %) -> P(>50K) on every test row, for the "disagree" filter


def probsOnTest(model, percent):
    if (model, percent) not in testProbs:
        testProbs[model, percent] = models[model][percent].predict_proba(testInputs[model])[:, 1]
    return testProbs[model, percent]


app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html", features=FEATURES, numeric=NUMERIC, categories=CATEGORIES, noisePercents=noisePercents,
    )


@app.get("/random")
def randomPerson():
    """A random test-set person; with disagree=1, only people the models disagree on"""
    percent = int(request.args.get("noise", 0))
    pool = np.arange(len(testDf))
    available = [model for model in MODELS if percent in models[model]]
    if request.args.get("disagree") == "1" and len(available) > 1:
        votes = sum(probsOnTest(model, percent) > 0.5 for model in available)  # models predicting >50K
        pool = pool[(votes > 0) & (votes < len(available))]
    person = testDf.iloc[np.random.default_rng().choice(pool)]
    values = {f: int(person[f]) if f in NUMERIC else person[f] for f in FEATURES}
    return jsonify(values | {"income": int(person["income"])})


@app.post("/predict")
def predict():
    """Run every model trained at the chosen noise level on the person in the request"""
    body = request.get_json()
    percent = int(body["noise"])
    person = pd.DataFrame([body["person"]])[FEATURES]
    person[NUMERIC] = person[NUMERIC].astype(int)
    errors = loadErrors()

    out = []
    for model, (name, transform) in MODELS.items():
        result = {"key": model, "name": name, "prob": None, "error": None}
        if percent in models[model]:
            # Encode the person into the model's input columns. Only the scaling, binning and category
            # codes are refit on the training set (same values as in the notebooks): the model is not retrained
            x = transform(trainDf, person)[1]
            result["prob"] = float(models[model][percent].predict_proba(x)[0, 1])
        if percent in errors.get(model, {}):
            result["error"] = float(errors[model][percent].mean())
        out.append(result)
    return jsonify(out)


@app.get("/errors")
def errorCurves():
    """Mean test error over the saved runs, per model and noise level, for the chart"""
    errors = loadErrors()
    return jsonify([
        {"key": model, "name": name, "points": [[int(p), float(e.mean())] for p, e in errors[model].items()]}
        for model, (name, _) in MODELS.items() if model in errors
    ])


if __name__ == "__main__":
    app.run()
