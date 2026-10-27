
import streamlit as st
import requests

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Fraud Detection Risk Scoring System")
st.write("Enter transaction details to assess fraud risk.")

st.info(
    "The prediction form must match the input fields "
    "expected by your trained fraud detection model."
)

st.subheader("API connection")
api_url = st.text_input(
    "FastAPI URL",
    placeholder="https://your-api-service-url"
)

if st.button("Check API health"):
    if not api_url.strip():
        st.warning("Enter your deployed FastAPI URL first.")
    else:
        try:
            response = requests.get(
                f"{api_url.rstrip('/')}/health",
                timeout=15
            )
            st.write("Status code:", response.status_code)
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")
