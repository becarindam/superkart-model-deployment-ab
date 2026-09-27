from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart")

MODEL_PATH = Path(__file__).resolve().parent / "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

NUMERIC_COLUMNS = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

CATEGORICAL_COLUMNS = [
    column for column in FEATURE_COLUMNS
    if column not in NUMERIC_COLUMNS
]


def validate_input(frame):
    """Check the model schema and numeric values before prediction."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURE_COLUMNS))

    if missing or extra:
        return None, {
            "error": "Input columns do not match the model schema.",
            "missing": missing,
            "extra": extra,
        }

    frame = frame[FEATURE_COLUMNS].copy()

    # Accept category labels used in the supplied batch CSV.
    frame["Product_Type_Category"] = (
        frame["Product_Type_Category"].replace({
            "Perishables": "Perishable",
            "Non Perishables": "Non-Perishable",
        })
    )

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not np.isfinite(frame[NUMERIC_COLUMNS].to_numpy(dtype=float)).all():
        return None, {"error": "Numeric fields must contain finite numbers."}

    if frame[CATEGORICAL_COLUMNS].isna().any().any():
        return None, {"error": "Categorical fields cannot be null."}

    if (frame[CATEGORICAL_COLUMNS].astype(str) == "").any().any():
        return None, {"error": "Categorical fields cannot be empty."}

    return frame, None


@superkart_api.get("/")
def home():
    return jsonify({
        "service": "SuperKart sales prediction",
        "status": "ok",
    })


@superkart_api.post("/v1/predict")
def predict_sales():
    record = request.get_json(silent=True)

    if not isinstance(record, dict):
        return jsonify({
            "error": "Send one product as a JSON object."
        }), 400

    input_frame, error = validate_input(pd.DataFrame([record]))
    if error:
        return jsonify(error), 400

    prediction = model.predict(input_frame)[0]
    return jsonify({"Sales": float(prediction)})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    uploaded_file = request.files.get("file")

    if uploaded_file is None:
        return jsonify({
            "error": "Upload a CSV using the 'file' field."
        }), 400

    try:
        batch_frame = pd.read_csv(uploaded_file)
    except (ValueError, pd.errors.ParserError, UnicodeError):
        return jsonify({
            "error": "The uploaded file is not a valid CSV."
        }), 400

    if batch_frame.empty:
        return jsonify({
            "error": "The CSV contains no product records."
        }), 400

    input_frame, error = validate_input(batch_frame)
    if error:
        return jsonify(error), 400

    predictions = model.predict(input_frame)

    return jsonify({
        str(index): round(float(value), 2)
        for index, value in enumerate(predictions)
    })


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
