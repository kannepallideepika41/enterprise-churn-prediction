import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn & CLV Engine", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('models/churn_xgb_pipeline.joblib')

model = load_model()

st.title("Enterprise Customer Churn & CLV Intelligence Simulator")
st.markdown("Predict individual customer churn probability, analyze risk indicators, and simulate expected lifetime value recovery.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile Inputs")
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_spend = st.number_input("Monthly Spend ($)", 20.0, 250.0, 75.0)
    total_charges = monthly_spend * tenure
    contract = st.selectbox("Contract Type", ['Month-to-Month', 'One-Year', 'Two-Year'])
    payment = st.selectbox("Payment Method", ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'])
    tickets = st.slider("Support Tickets (Past 30 Days)", 0, 10, 2)
    usage_drop = st.slider("Usage Drop Ratio (-1.0 to 1.0)", -1.0, 1.0, -0.2)
    failed_pay = st.slider("Failed Payments (Last 6 Months)", 0, 5, 1)

with col2:
    st.subheader("Inference & Retention Action")
    
    spend_ratio = monthly_spend / (tenure + 1)
    risk_index = (tickets * 1.5) + (failed_pay * 2.0)
    
    input_data = pd.DataFrame([{
        'tenure_months': tenure,
        'monthly_spend': monthly_spend,
        'total_charges': total_charges,
        'contract_type': contract,
        'payment_method': payment,
        'support_tickets_30d': tickets,
        'usage_drop_ratio': usage_drop,
        'failed_payments_6m': failed_pay,
        'spend_to_tenure_ratio': spend_ratio,
        'risk_score_index': risk_index
    }])
    
    churn_prob = model.predict_proba(input_data)[0][1]
    clv_baseline = monthly_spend * 24
    expected_clv = clv_baseline * (1 - churn_prob)
    
    st.metric(label="Churn Probability", value=f"{churn_prob:.1%}")
    st.metric(label="Baseline 24-Mo Value", value=f"${clv_baseline:,.2f}")
    st.metric(label="Risk-Adjusted Expected CLV", value=f"${expected_clv:,.2f}")
    
    if churn_prob > 0.65:
        st.error("HIGH CHURN RISK: Trigger proactive outreach, contract discount, or account manager check-in.")
    elif churn_prob > 0.35:
        st.warning("MODERATE CHURN RISK: Recommend service optimization check.")
    else:
        st.success("LOW CHURN RISK: Healthy account status.")