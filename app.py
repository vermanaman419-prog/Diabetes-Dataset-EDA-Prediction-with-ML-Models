# app.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.exceptions import NotFittedError

# ---- Configuration ----
MODEL_FILENAME = os.environ.get("MODEL_FILE", "xgboost_model.joblib")
# Use the PORT provided by the environment (Hugging Face / Render / similar)
HF_PORT = int(os.environ.get("PORT", 7860))

# Replace these with the exact feature names & order your model expects.
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ---- Robust model loader ----
def load_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Model file not found at: {path}", file=sys.stderr)
        return None
    try:
        m = joblib.load(path)
        print(f"[INFO] Loaded model from {path}", file=sys.stderr)
        return m
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        return None

model = load_model(MODEL_FILENAME)

# ---- Input builder ----
def build_input_df(values):
    """
    values: dict mapping FEATURE_NAMES -> values
    returns: single-row DataFrame with numeric dtypes and columns in FEATURE_NAMES order
    """
    row = {k: values.get(k, 0) for k in FEATURE_NAMES}
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    # coerce to numeric and fill NaN
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

# ---- Prediction function ----
def predict(
    Pregnancies, Glucose, BloodPressure, SkinThickness,
    Insulin, BMI, DiabetesPedigreeFunction, Age
):
    if model is None:
        return "Model not loaded. Upload the joblib file and redeploy."

    values = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    X = build_input_df(values)

    try:
        pred = model.predict(X)[0]
    except NotFittedError:
        return "Model is not fitted. Please re-save a fitted model."
    except Exception as e:
        return f"Prediction failed: {e}"

    prob_text = ""
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            # If binary, positive class probability is index 1
            if len(p) >= 2:
                prob = float(p[1])
            else:
                prob = float(p[0])
            prob_text = f"\nProbability (positive): {prob:.3f}"
    except Exception:
        prob_text = ""

    label = "🔴 Positive" if int(pred) == 1 else "🟢 Negative"
    preview = X.to_dict(orient="records")[0]
    return f"{label}{prob_text}\n\nInput: {preview}"

# ---- Gradio UI ----
title = "Diabetes Prediction (Random Forest)"
description = "Enter numeric patient features and click Predict. Make sure your model expects these features in the shown order."

inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="BloodPressure", value=70),
    gr.Number(label="SkinThickness", value=20),
    gr.Number(label="Insulin", value=79),
    gr.Number(label="BMI", value=30.0),
    gr.Number(label="DiabetesPedigreeFunction", value=0.5),
    gr.Number(label="Age", value=33),
]

# A simple example list (optional) — remove or update if you want
examples = [
    [2, 120, 70, 20, 79, 30.0, 0.5, 33],
    [6, 140, 90, 30, 130, 35.5, 0.65, 48],
]

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column(scale=1):
            comps = []
            for comp in inputs:
                comps.append(comp)
                comp.render()
            btn = gr.Button("Predict")
        with gr.Column(scale=1):
            out = gr.Textbox(label="Prediction Result", lines=6)
    btn.click(fn=predict, inputs=inputs, outputs=out)
    # Examples block (optional)
    gr.Examples(examples=examples, inputs=inputs)

# ---- Launch (bind to 0.0.0.0 and PORT so Spaces works) ----
if __name__ == "__main__":
    # Use server_name and server_port so the environment can expose the app.
    demo.launch(server_name="0.0.0.0", server_port=HF_PORT, share=False)

