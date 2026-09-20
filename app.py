import streamlit as st
import joblib
import pandas as pd
import numpy as np


# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "model/rf_augmented.pkl"

model = joblib.load(MODEL_PATH)


# ============================================
# FEATURES
# ============================================

selected_features = [
    'step',
    'amount',
    'log_amount',
    'positive_depletion',
    'log_depletion',
    'drain_ratio_capped',
    'remaining_ratio_capped',
    'balance_after_transaction',
    'receiver_balance_change',
    'log_receiver_balance_change',
    'receiver_previous_transaction_count',
    'receiver_is_new',
    'type_CASH_IN',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER'
]


# ============================================
# FEATURE ENGINEERING
# ============================================

def prepare_transaction(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    receiver_previous_transaction_count
):

    log_amount = np.log1p(amount)

    positive_depletion = max(
        oldbalanceOrg - newbalanceOrig,
        0
    )

    log_depletion = np.log1p(
        positive_depletion
    )

    if oldbalanceOrg > 0:

        drain_ratio_capped = np.clip(
            amount / oldbalanceOrg,
            0,
            1
        )

        remaining_ratio_capped = np.clip(
            newbalanceOrig / oldbalanceOrg,
            0,
            1
        )

    else:

        drain_ratio_capped = 0
        remaining_ratio_capped = 0

    balance_after_transaction = (
        oldbalanceOrg - amount
    )

    receiver_balance_change = (
        newbalanceDest - oldbalanceDest
    )

    log_receiver_balance_change = np.log1p(
        max(receiver_balance_change, 0)
    )

    receiver_is_new = int(
        receiver_previous_transaction_count == 0
    )

    data = {
        'step': step,
        'amount': amount,
        'log_amount': log_amount,
        'positive_depletion': positive_depletion,
        'log_depletion': log_depletion,
        'drain_ratio_capped': drain_ratio_capped,
        'remaining_ratio_capped': remaining_ratio_capped,
        'balance_after_transaction':
            balance_after_transaction,
        'receiver_balance_change':
            receiver_balance_change,
        'log_receiver_balance_change':
            log_receiver_balance_change,
        'receiver_previous_transaction_count':
            receiver_previous_transaction_count,
        'receiver_is_new':
            receiver_is_new,
        'type_CASH_IN':
            int(transaction_type == "CASH_IN"),
        'type_CASH_OUT':
            int(transaction_type == "CASH_OUT"),
        'type_DEBIT':
            int(transaction_type == "DEBIT"),
        'type_PAYMENT':
            int(transaction_type == "PAYMENT"),
        'type_TRANSFER':
            int(transaction_type == "TRANSFER")
    }

    X = pd.DataFrame([data])

    return X[selected_features]


# ============================================
# PAGE
# ============================================

st.set_page_config(
    page_title="PaySim Fraud Detection",
    page_icon="🔐",
    layout="centered"
)


st.title("🔐 Mobile Money Fraud Detection")

st.write(
    "Machine learning-based detection of "
    "potentially suspicious mobile-money transactions."
)


# ============================================
# INPUTS
# ============================================

st.subheader("Transaction Information")

step = st.number_input(
    "Transaction step",
    min_value=1,
    value=500
)

transaction_type = st.selectbox(
    "Transaction type",
    [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]
)

amount = st.number_input(
    "Transaction amount",
    min_value=0.0,
    value=10000.0
)

oldbalanceOrg = st.number_input(
    "Sender balance before transaction",
    min_value=0.0,
    value=500000.0
)

newbalanceOrig = st.number_input(
    "Sender balance after transaction",
    min_value=0.0,
    value=490000.0
)

oldbalanceDest = st.number_input(
    "Receiver balance before transaction",
    min_value=0.0,
    value=100000.0
)

newbalanceDest = st.number_input(
    "Receiver balance after transaction",
    min_value=0.0,
    value=110000.0
)

receiver_previous_transaction_count = st.number_input(
    "Receiver previous transaction count",
    min_value=0,
    value=20
)


# ============================================
# PREDICTION
# ============================================

if st.button("Analyze Transaction"):

    X = prepare_transaction(
        step,
        transaction_type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        receiver_previous_transaction_count
    )

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0, 1]

    st.subheader("Detection Result")

    if prediction == 1:

        st.error("⚠️ SUSPICIOUS TRANSACTION")

    else:

        st.success("✅ NORMAL TRANSACTION")

    st.write(
        f"Fraud probability: "
        f"{probability:.4%}"
    )