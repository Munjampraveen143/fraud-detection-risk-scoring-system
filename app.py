
import streamlit as st
import requests

st.set_page_config(
    page_title="Fraud Detection Risk Scoring",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Detection Risk Scoring System")
st.write("Enter transaction details to assess fraud risk.")

# API configuration: Streamlit Cloud Secrets first,
# with a sidebar override available for testing.
configured_api_url = st.secrets.get("API_URL", "")

st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input(
    "FastAPI base URL",
    value=configured_api_url,
    placeholder="https://your-api-service.onrender.com",
    help="Enter your deployed FastAPI URL, without /predict or /health.",
).strip().rstrip("/")

if api_url.startswith("http://127.0.0.1") or api_url.startswith(
    "http://localhost"
):
    st.sidebar.warning(
        "A local URL only works when the API is accessible "
        "from the same computer. Streamlit Cloud needs a public API URL."
    )

st.subheader("Transaction Details")

with st.form("fraud_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction amount",
            min_value=0.01,
            value=100.00,
            step=10.0,
        )
        transaction_hour = st.number_input(
            "Transaction hour (0–23)",
            min_value=0,
            max_value=23,
            value=14,
        )
        transactions_last_hour = st.number_input(
            "Transactions in the last hour",
            min_value=0,
            value=1,
        )
        account_age_days = st.number_input(
            "Account age (days)",
            min_value=0,
            value=365,
        )

    with col2:
        previous_failed_transactions = st.number_input(
            "Previous failed transactions",
            min_value=0,
            value=0,
        )
        device_changed = st.selectbox(
            "Device changed?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )
        location_changed = st.selectbox(
            "Location changed?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )

    submitted = st.form_submit_button(
        "Assess Fraud Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not api_url:
        st.error(
            "FastAPI URL is missing. Deploy your API and set "
            "API_URL in Streamlit Secrets."
        )
    else:
        payload = {
            "amount": amount,
            "transaction_hour": transaction_hour,
            "transactions_last_hour": transactions_last_hour,
            "account_age_days": account_age_days,
            "previous_failed_transactions": previous_failed_transactions,
            "device_changed": device_changed,
            "location_changed": location_changed,
        }

        try:
            with st.spinner("Assessing transaction risk..."):
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=30,
                )

            if response.ok:
                result = response.json()
                decision = result.get("decision", "UNKNOWN")
                score = result.get("risk_score")
                probability = result.get("fraud_probability")

                st.subheader("Prediction Result")

                if decision == "BLOCK":
                    st.error("🚨 BLOCK — High predicted fraud risk")
                elif decision == "REVIEW":
                    st.warning("⚠️ REVIEW — Further review recommended")
                elif decision == "APPROVE":
                    st.success("✅ APPROVE — Low predicted fraud risk")
                else:
                    st.info(f"Decision: {decision}")

                if score is not None:
                    score = float(score)
                    st.metric("Fraud Risk Score", f"{score:.2f}%")
                    st.progress(max(0.0, min(1.0, score / 100)))

                if probability is not None:
                    st.caption(
                        f"Model fraud probability: "
                        f"{float(probability):.2%}"
                    )

                database_status = result.get("database_status")
                if database_status == "saved":
                    st.success("Transaction prediction saved to database.")
                elif database_status is not None:
                    st.info(f"Database status: {database_status}")
                else:
                    st.caption("The API did not return a database status.")

            else:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", response.text)
                except ValueError:
                    detail = response.text

                st.error(
                    f"Prediction failed ({response.status_code}): {detail}"
                )

        except requests.Timeout:
            st.error(
                "The API took too long to respond. Try again or check "
                "whether your backend is running."
            )
        except requests.RequestException as exc:
            st.error(
                "Could not reach the prediction API. Check the public URL "
                "and ensure the FastAPI backend is running."
            )
            st.caption(str(exc))

st.divider()
st.caption(
    "This is a model-generated risk assessment, not a guarantee "
    "that a transaction is fraudulent or legitimate."
)
