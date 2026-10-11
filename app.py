
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import xgboost as xgb
from sklearn.pipeline import Pipeline

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("Used Car Price Prediction")
#st.write("Compare predictions from six machine learning regression models.")

# ---------------- LOAD MODELS ----------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")



MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "LightGBM": "lightgbm.pkl",
    "CatBoost": "catboost.pkl",
    "LightGBM (Log Target)": "lightgbm_log.pkl",
    "Tuned LightGBM (Log Target)": "tuned_lightgbm_log.pkl",
}

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded

try:
    models = load_models()
except Exception as e:
    st.error(f"Could not load a model: {e}")
    st.stop()


# Load XGBoost using its native JSON model and saved preprocessor
try:
   
# ---------------- LOAD XGBOOST ----------------
    xgb_path = os.path.join(MODEL_DIR, "xgboost.pkl")
    if os.path.exists(xgb_path):
        try:
            models["XGBoost"] = joblib.load(xgb_path)
            st.success("XGBoost model loaded successfully.")
        except Exception as e:
            st.warning(f"XGBoost could not be loaded: {e}")
    else:
        st.warning(
            "XGBoost model file not found. "
            "Expected: models/xgboost.pkl"
        )


except Exception as e:
    st.warning(f"XGBoost could not be loaded: {e}")


if not models:
    st.error("No model files found. Check that the models folder is beside app.py.")
    st.stop()

st.success(f"Loaded {len(models)} of 6 models.")

# ---------------- CAR DETAILS ----------------
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz",
         "Hyundai", "Audi", "Chevrolet", "Kia", "Nissan", "Other"]
    )

    model_name = st.text_input("Car Model", value="Camry")

    model_year = st.selectbox(
        "Model Year",
        list(range(2026, 1999, -1)),
        index=10
    )

    milage = st.number_input(
        "Mileage (miles)",
        min_value=0,
        max_value=1_000_000,
        value=30000,
        step=1000
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Gasoline", "Hybrid", "Diesel", "E85 Flex Fuel",
         "Plug-In Hybrid", "Electric", "Unknown"]
    )

with col2:
    engine = st.text_input(
        "Engine Description",
        value="2.5L I4"
    )

    transmission = st.selectbox(
        "Transmission",
        ["Automatic", "A/T", "8-Speed Automatic", "6-Speed Automatic",
         "Manual", "CVT", "Other"]
    )

    ext_col = st.selectbox(
        "Exterior Color",
        ["Black", "White", "Gray", "Silver", "Blue", "Red", "Other"]
    )

    int_col = st.selectbox(
        "Interior Color",
        ["Black", "Beige", "Gray", "Brown", "Other"]
    )

    accident = st.selectbox(
        "Accident History",
        ["None reported", "At least 1 accident or damage reported", "Unknown"]
    )

    clean_title = st.selectbox(
        "Clean Title",
        ["Yes", "No", "Unknown"]
    )

# ---------------- PREPARE INPUT ----------------
input_data = pd.DataFrame([{
    "brand": brand,
    "model": model_name,
    "model_year": model_year,
    "milage": milage,
    "fuel_type": fuel_type,
    "engine": engine,
    "transmission": transmission,
    "ext_col": ext_col,
    "int_col": int_col,
    "accident": accident,
    "clean_title": clean_title
}])

# ---------------- PREDICT ----------------
# if st.button("Compare Model Predictions", type="primary"):
#     results = []

#     for name, model in models.items():
#         try:
#             prediction = float(model.predict(input_data)[0])
#             results.append({
#                 "Model": name,
#                 "Predicted Price ($)": max(0, prediction)
#             })
#         except Exception as e:
#             st.warning(f"{name} could not predict this car: {e}")

#     if results:
#         results_df = pd.DataFrame(results).sort_values(
#             "Predicted Price ($)", ascending=False
#         )

#         st.subheader("Predicted Prices")
#         st.dataframe(
#             results_df.style.format({"Predicted Price ($)": "${:,.2f}"}),
#             use_container_width=True,
#             hide_index=True
#         )

#         fig = px.bar(
#             results_df,
#             x="Model",
#             y="Predicted Price ($)",
#             title="Predicted Price by Model",
#             text_auto=".2s"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         st.info(
#             "These are estimates from different models. "
#             "The highest predicted price does not mean that model is the most accurate."
#         )

# ---------------- PREDICT ----------------
prediction_mode = st.radio(
    "Prediction Mode",
    [
        "Predict with Tuned Model",
        "Compare All Models"
    ],
    horizontal=True
)

if st.button("Predict Car Price", type="primary"):
    if prediction_mode == "Predict with Tuned Model":
        model_name = "Tuned LightGBM (Log Target)"

        if model_name not in models:
            st.error(
                "Tuned model not found. Check that "
                "models/tuned_lightgbm_log.pkl exists."
            )
        else:
            try:
                prediction = float(
                    models[model_name].predict(input_data)[0]
                )
                prediction = max(0, prediction)

                st.subheader("Estimated Car Price")
                st.metric(
                    label="Predicted Price",
                    value=f"${prediction:,.2f}"
                )
                st.caption(
                    "This is an estimate, not a guaranteed selling price."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        results = []

        for name, model in models.items():
            try:
                prediction = float(model.predict(input_data)[0])

                results.append({
                    "Model": name,
                    "Predicted Price ($)": max(0, prediction)
                })

            except Exception as e:
                st.warning(f"{name} could not predict: {e}")

        if results:
            results_df = pd.DataFrame(results).sort_values(
                "Predicted Price ($)",
                ascending=False
            )

            st.subheader("Predicted Prices")
            st.dataframe(
                results_df.style.format({
                    "Predicted Price ($)": "${:,.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                results_df,
                x="Model",
                y="Predicted Price ($)",
                title="Predicted Price by Model",
                text_auto=".2s"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "These are estimates from different models. "
                "The highest predicted price does not mean that "
                "model is the most accurate."
            )

# ---------------- PERFORMANCE EXPLANATION ----------------
# st.divider()
# st.subheader("How Accurate Are the Models?")


# st.write(
#     "Estimate a used car's price using the tuned LightGBM model "
#     "or compare predictions from multiple machine learning models."
# )


