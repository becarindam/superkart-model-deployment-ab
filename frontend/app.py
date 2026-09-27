import io
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL", "http://backend:7860"
).rstrip("/")

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    layout="wide",
)

st.title("SuperKart Sales Prediction")
st.write(
    "Estimate sales for a product–store record using the trained model."
)

st.subheader("Single Prediction")

with st.form("single_prediction_form"):
    left, right = st.columns(2)

    with left:
        product_weight = st.number_input(
            "Product weight",
            min_value=0.0,
            value=12.66,
        )
        sugar_content = st.selectbox(
            "Product sugar content",
            ["Low Sugar", "Regular", "No Sugar"],
        )
        allocated_area = st.number_input(
            "Product allocated area ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.069,
            format="%.3f",
        )
        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=147.03,
        )
        product_prefix = st.selectbox(
            "Product ID prefix",
            ["FD", "DR", "NC"],
        )

    with right:
        store_size = st.selectbox(
            "Store size",
            ["Medium", "High", "Small"],
        )
        city_type = st.selectbox(
            "Store location city type",
            ["Tier 2", "Tier 1", "Tier 3"],
        )
        store_type = st.selectbox(
            "Store type",
            [
                "Supermarket Type2",
                "Supermarket Type1",
                "Departmental Store",
                "Food Mart",
            ],
        )
        store_age = st.number_input(
            "Store age in years (2025 reference)",
            min_value=0,
            value=16,
            step=1,
        )
        product_type_category = st.selectbox(
            "Product type category",
            ["Perishable", "Non-Perishable"],
        )

    submit_single = st.form_submit_button(
        "Predict sales",
        type="primary",
    )

if submit_single:
    record = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_prefix,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_type_category,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=record,
            timeout=20,
        )
        response.raise_for_status()
        predicted_sales = response.json()["Sales"]
        st.success(f"Predicted sales: {predicted_sales:,.2f}")
    except (requests.RequestException, KeyError, ValueError) as error:
        st.error(f"Prediction failed: {error}")


st.divider()
st.subheader("Batch Prediction")
st.write(
    "Upload a CSV with the ten model input columns. "
    "The supplied Batch_Data_SuperKart.csv is an example."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
)

if uploaded_file is not None and st.button(
    "Predict batch",
    type="primary",
):
    csv_bytes = uploaded_file.getvalue()

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={
                "file": (
                    uploaded_file.name,
                    csv_bytes,
                    "text/csv",
                )
            },
            timeout=60,
        )
        response.raise_for_status()

        predictions = response.json()
        batch_data = pd.read_csv(io.BytesIO(csv_bytes))
        batch_data["Predicted_Sales"] = [
            predictions[str(index)]
            for index in range(len(batch_data))
        ]

        st.success(
            f"Predictions completed for {len(batch_data)} records."
        )
        st.dataframe(batch_data, use_container_width=True)

        st.download_button(
            "Download predictions as CSV",
            data=batch_data.to_csv(index=False).encode("utf-8"),
            file_name="superkart_batch_predictions.csv",
            mime="text/csv",
        )
    except (
        requests.RequestException,
        KeyError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Batch prediction failed: {error}")
