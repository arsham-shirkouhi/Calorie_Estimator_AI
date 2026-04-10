from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.predict import get_nutrition, predict_food


def _format_food_name(food_class: str) -> str:
    return food_class.replace("_", " ").title()


def _build_nutrition_table(nutrition: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Nutrient": ["Calories", "Protein", "Carbs", "Fat", "Fiber", "Sugar", "Sodium"],
            "Per 100g": [
                nutrition["calories"],
                nutrition["protein"],
                nutrition["carbs"],
                nutrition["fat"],
                nutrition["fiber"],
                nutrition["sugar"],
                nutrition["sodium"],
            ],
        }
    )


st.set_page_config(page_title="Calorie Estimator AI", layout="wide")

st.title("Calorie Estimator AI")
st.write(
    "Upload a food photo and the app will predict the dish, show its confidence score, "
    "and display estimated nutrition values per 100g."
)

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload a JPG or PNG image to get a prediction and nutrition estimate.")
else:
    image_col, results_col = st.columns([1, 1.1], gap="large")

    with image_col:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    with results_col:
        st.subheader("Results")

        if not MODEL_PATH.exists():
            st.error(
                "The trained model file could not be found. Please make sure "
                "`models/best_model.pth` is available before running predictions."
            )
        else:
            temp_path: Path | None = None

            try:
                suffix = Path(uploaded_file.name).suffix or ".jpg"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = Path(temp_file.name)

                with st.spinner("Analyzing your image..."):
                    predicted_class, confidence = predict_food(temp_path, model_path=MODEL_PATH)
                    nutrition = get_nutrition(predicted_class)

                prediction_col, confidence_col = st.columns(2)
                prediction_col.metric("Predicted Food", _format_food_name(predicted_class))
                confidence_col.metric("Confidence", f"{confidence:.2f}%")

                st.subheader("Nutrition per 100g")
                st.dataframe(_build_nutrition_table(nutrition), hide_index=True, use_container_width=True)
                st.caption("These values come from the nutrition database and are shown per 100g.")

            except ValueError as error:
                if "not found in nutrition database" in str(error):
                    st.error(
                        "The predicted food class was not found in the nutrition database. "
                        "Please update the dataset or try a different image."
                    )
                else:
                    st.error(f"Could not process this image: {error}")
            except FileNotFoundError:
                st.error(
                    "A required app file is missing. Please check that the model and data files "
                    "are available, then try again."
                )
            except Exception:
                st.error(
                    "Something went wrong while analyzing the image. Please try again with a "
                    "different image."
                )
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
