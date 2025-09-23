from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import requests
import streamlit as st


st.set_page_config(page_title="Personality Classifier", page_icon="🤖", layout="centered")

st.title("Personality Classifier")
st.caption(
    "This tool predicts whether your personality leans more Introvert or Extrovert based on everyday habits. "
    "Adjust the sliders and inputs below to describe yourself, then press Predict."
)


# Internal API address (not shown in the UI)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


st.subheader("Prediction Inputs")

col1, col2 = st.columns(2)
with col1:
    # 0–10 plus "10+" (mapped to 11 for backend)
    _tsa_options = list(range(0, 11)) + ["10+"]
    _tsa_value = st.select_slider(
        label="Time spent alone (hours per day)",
        options=_tsa_options,
        value=2,
        help="Maps to backend: Time_spent_Alone (""10+"" becomes 11)",
    )
    time_spent_alone = 11.0 if _tsa_value == "10+" else float(_tsa_value)
    social_event_attendance = st.slider(
        label="Social event attendance (0–10)",
        min_value=0,
        max_value=10,
        value=6,
        help="Maps to backend field: Social_event_attendance",
    )
    going_outside = st.number_input(
        label="Days you go outside per week (0–7)",
        min_value=0,
        max_value=7,
        value=3,
        step=1,
        help="Maps to backend field: Going_outside",
    )
with col2:
    # Integers only: 0–14 plus "14+" (mapped to 15 for backend)
    _fcs_options = list(range(0, 15)) + ["14+"]
    _fcs_value = st.select_slider(
        label="Friends circle size (people)",
        options=_fcs_options,
        value=10,
        help="Maps to backend: Friends_circle_size (""14+"" becomes 15)",
    )
    friends_circle_size = 15.0 if _fcs_value == "14+" else float(_fcs_value)
    post_frequency = st.slider(
        label="Social media posting frequency (0–10)",
        min_value=0,
        max_value=10,
        value=5,
        help="Maps to backend field: Post_frequency",
    )
    stage_fear_human = st.selectbox(
        label="Stage fright",
        options=["No", "Yes"],
        index=0,
        help="Maps to backend field: Stage_fear (Yes/No)",
    )
    drained_after_human = st.selectbox(
        label="Feel drained after socializing",
        options=["No", "Yes"],
        index=0,
        help="Maps to backend field: Drained_after_socializing (Yes/No)",
    )


def to_backend_yes_no(value: str) -> str:
    return "Yes" if value == "Yes" else "No"


predictions: Union[List[Any], None] = None
error_message: str | None = None

if st.button("Predict"):
    try:
        payload: Dict[str, Any] = {
            "samples": {
                "Time_spent_Alone": float(time_spent_alone),
                "Social_event_attendance": float(social_event_attendance),
                "Going_outside": float(going_outside),
                "Friends_circle_size": float(friends_circle_size),
                "Post_frequency": float(post_frequency),
                "Stage_fear": to_backend_yes_no(stage_fear_human),
                "Drained_after_socializing": to_backend_yes_no(drained_after_human),
            }
        }

        with st.spinner("Running prediction..."):
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        if resp.ok:
            data = resp.json()
            predictions = data.get("predictions")
        else:
            error_message = f"API error {resp.status_code}: {resp.text}"
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)


st.subheader("Prediction Result")
if predictions is not None:
    # Model returns 0/1. Map to labels.
    def to_label(x: Any) -> str:
        try:
            return "Extrovert" if int(x) == 1 else "Introvert"
        except Exception:  # noqa: BLE001
            return str(x)

    if isinstance(predictions, list) and len(predictions) == 1:
        label = to_label(predictions[0])
        st.success(f"Prediction: {label}")
    else:
        labeled = [to_label(x) for x in (predictions if isinstance(predictions, list) else [predictions])]
        st.success("Predictions received")
        st.write(labeled)
elif error_message:
    st.error(error_message)
else:
    st.info("Set your inputs above and click Predict to see your result here.")

st.markdown("---")
