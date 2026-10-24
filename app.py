import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# 1. APP CONFIGURATION AND LOAD MODEL FILES
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "log_model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"
FEATURES_PATH = APP_DIR / "features"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, scaler, list(features)


try:
    log_model, scaler, features = load_artifacts()

except FileNotFoundError as exc:
    st.error(
        f"Required model file not found: {exc.filename}. "
        "Ensure log_model.pkl, scaler.pkl, and features "
        "are in the same folder as app.py."
    )
    st.stop()

except Exception as exc:
    st.error(f"Unable to load model files: {exc}")
    st.stop()


# ---------------------------------------------------------
# 2. DEFINE NUMERICAL AND BINARY COLUMNS
# ---------------------------------------------------------

NUM_COLS = [
    "monthly_charges",
    "total_charges",
    "satisfaction_score",
    "avg_monthly_value",
    "num_support_tickets",
    "charges_per_ticket",
    "engagement_score",
    "tenure_months"
]

CATEGORICAL_COLS = [
    "contract_type",
    "internet_service",
    "payment_method",
    "tenure_group"
]

BINARY_COLS = [
    "tech_support",
    "streaming_service",
    "paperless_billing"
]


# ---------------------------------------------------------
# 3. GET CATEGORIES FROM SAVED FEATURE NAMES
# ---------------------------------------------------------

def get_category_options(prefix, saved_features):
    marker = f"{prefix}_"

    options = [
        feature[len(marker):]
        for feature in saved_features
        if feature.startswith(marker)
    ]

    return sorted(set(options))


contract_options = get_category_options(
    "contract_type", features
)

internet_options = get_category_options(
    "internet_service", features
)

payment_options = get_category_options(
    "payment_method", features
)


# Fallback categories if the saved feature list
# does not contain the expected one-hot columns.

if not contract_options:
    contract_options = [
        "Month-to-month",
        "One year",
        "Two year"
    ]

if not internet_options:
    internet_options = [
        "DSL",
        "Fiber optic",
        "No"
    ]

if not payment_options:
    payment_options = [
        "Bank transfer",
        "Credit card",
        "Electronic check",
        "Mailed check"
    ]


# ---------------------------------------------------------
# 4. FEATURE ENGINEERING AND PREPROCESSING
# ---------------------------------------------------------

def build_model_input(customer):

    row = pd.DataFrame([customer])

    # Ensure numerical columns are float to prevent int64 assignment errors during scaling
    for col in NUM_COLS:
        if col in row.columns:
            row[col] = row[col].astype(float)

    # Average monthly customer value
    row["avg_monthly_value"] = (
        row["total_charges"] / row["tenure_months"]
    )

    # Tenure segmentation
    def tenure_group(tenure):

        if tenure <= 12:
            return "new"

        elif tenure <= 36:
            return "mid"

        else:
            return "old"

    row["tenure_group"] = (
        row["tenure_months"].apply(tenure_group)
    )

    # High-risk customer flag
    row["high_risk"] = (
        (row["satisfaction_score"] <= 2)
        & (row["num_support_tickets"] >= 3)
    ).astype(int)

    # Charges per support ticket
    row["charges_per_ticket"] = (
        row["monthly_charges"]
        / (row["num_support_tickets"] + 1)
    )

    # Engagement score
    row["engagement_score"] = (
        row["tenure_months"] * 0.4
        + row["satisfaction_score"] * 0.3
        - row["num_support_tickets"] * 0.3
    )

    # Ensure binary columns are integers: 0 or 1
    for col in BINARY_COLS:
        row[col] = row[col].astype(int)

    # One-hot encode categorical columns
    row = pd.get_dummies(
        row,
        columns=CATEGORICAL_COLS,
        dtype=int
    )

    # Align input columns with training features
    row = row.reindex(
        columns=features,
        fill_value=0
    )

    # Check required numerical columns
    missing_num_cols = [
        col for col in NUM_COLS
        if col not in row.columns
    ]

    if missing_num_cols:
        raise ValueError(
            "Missing numerical columns in saved features: "
            + ", ".join(missing_num_cols)
        )

    # Scale numerical columns using the saved scaler
    row.loc[:, NUM_COLS] = scaler.transform(
        row[NUM_COLS]
    )

    # Ensure exact training feature order
    row = row.reindex(
        columns=features,
        fill_value=0
    )

    return row


# ---------------------------------------------------------
# 5. STREAMLIT APP INTERFACE
# ---------------------------------------------------------

st.title("Customer Churn Prediction")

st.write(
    "Enter a customer's details to estimate their "
    "likelihood of leaving the company."
)

st.info(
    "This application uses a trained Logistic Regression "
    "model to identify customers who may be at risk of churn."
)


with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        tenure_months = st.number_input(
            "Tenure (Months)",
            min_value=1,
            max_value=600,
            value=12,
            step=1
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.01,
            value=50.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=600.0,
            step=10.0
        )

        satisfaction_score = st.slider(
            "Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3
        )

        num_support_tickets = st.number_input(
            "Number of Support Tickets",
            min_value=0,
            max_value=1000,
            value=1,
            step=1
        )

    with col2:

        contract_type = st.selectbox(
            "Contract Type",
            options=contract_options
        )

        internet_service = st.selectbox(
            "Internet Service",
            options=internet_options
        )

        payment_method = st.selectbox(
            "Payment Method",
            options=payment_options
        )

        tech_support = st.selectbox(
            "Tech Support",
            options=["Yes", "No"]
        )

        streaming_service = st.selectbox(
            "Streaming Service",
            options=["Yes", "No"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=["Yes", "No"]
        )

    st.subheader("Prediction Settings")

    threshold = st.slider(
        "Churn-Risk Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Lower thresholds generally identify more potential "
            "churners but may also increase false positives."
        )
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# 6. MAKE PREDICTION
# ---------------------------------------------------------

if submitted:

    if (
        total_charges < 0
        or monthly_charges <= 0
        or tenure_months < 1
    ):

        st.error(
            "Please enter valid customer information."
        )

        st.stop()

    # Convert user inputs into model-ready values
    customer = {

        "tenure_months": int(tenure_months),

        "monthly_charges": float(
            monthly_charges
        ),

        "total_charges": float(
            total_charges
        ),

        "contract_type": contract_type,

        "internet_service": internet_service,

        "tech_support": (
            1 if tech_support == "Yes" else 0
        ),

        "streaming_service": (
            1 if streaming_service == "Yes" else 0
        ),

        "payment_method": payment_method,

        "paperless_billing": (
            1 if paperless_billing == "Yes" else 0
        ),

        "satisfaction_score": int(
            satisfaction_score
        ),

        "num_support_tickets": int(
            num_support_tickets
        )
    }

    try:

        # Prepare features
        model_input = build_model_input(
            customer
        )

        # Predict probability of churn
        churn_probability = float(
            log_model.predict_proba(
                model_input
            )[0, 1]
        )

        # Apply selected threshold
        predicted_churn = int(
            churn_probability >= threshold
        )

        st.divider()

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if predicted_churn == 1:

                st.error(
                    "Customer is AT RISK of churn"
                )

            else:

                st.success(
                    "Customer is NOT flagged as at risk"
                )

        with result_col2:

            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.1%}"
            )

        # Visual probability indicator
        st.progress(churn_probability)

        st.write(
            f"**Selected threshold:** {threshold:.0%}"
        )

        if predicted_churn == 1:

            st.warning(
                "Recommended action: review the customer's "
                "experience and consider a retention intervention, "
                "such as contacting the customer to understand "
                "their concerns."
            )

        else:

            st.info(
                "The estimated churn probability is below the "
                "selected threshold. Continue normal monitoring."
            )

        # Optional model input inspection
        with st.expander(
            "View Engineered Features Sent to the Model"
        ):

            st.dataframe(
                model_input,
                use_container_width=True
            )

    except Exception as exc:

        st.error(
            f"Prediction could not be completed: {exc}"
        )

        st.caption(
            "Verify that the model, scaler and feature list "
            "were saved from the same training workflow."
        )