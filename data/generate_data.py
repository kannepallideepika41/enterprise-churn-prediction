import numpy as np
import pandas as pd
import os

def generate_enterprise_churn_data(n_records=70000, random_state=42):
    np.random.seed(random_state)
    
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_records + 1)]
    tenure_months = np.random.randint(1, 72, size=n_records)
    monthly_spend = np.random.uniform(20.0, 180.0, size=n_records)
    total_charges = monthly_spend * tenure_months + np.random.normal(0, 15, size=n_records)
    total_charges = np.maximum(total_charges, monthly_spend)
    
    contract_types = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_records, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'], size=n_records)
    support_tickets_30d = np.random.poisson(lam=1.2, size=n_records)
    usage_drop_ratio = np.clip(np.random.normal(0.0, 0.35, size=n_records), -1.0, 1.0)
    failed_payments_6m = np.random.poisson(lam=0.4, size=n_records)
    
    # Calculate churn propensity log-odds
    log_odds = (
        -1.8
        + 0.035 * monthly_spend
        - 0.05 * tenure_months
        + 0.45 * support_tickets_30d
        + 0.65 * failed_payments_6m
        + 1.2 * (contract_types == 'Month-to-Month')
        - 0.8 * (contract_types == 'Two-Year')
        + 1.5 * np.maximum(0, -usage_drop_ratio)
    )
    probs = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.uniform(0, 1, size=n_records) < probs).astype(int)
    
    # Calculate Customer Lifetime Value (CLV)
    estimated_clv = (monthly_spend * 24 * (1 - probs)).round(2)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure_months': tenure_months,
        'monthly_spend': np.round(monthly_spend, 2),
        'total_charges': np.round(total_charges, 2),
        'contract_type': contract_types,
        'payment_method': payment_method,
        'support_tickets_30d': support_tickets_30d,
        'usage_drop_ratio': np.round(usage_drop_ratio, 3),
        'failed_payments_6m': failed_payments_6m,
        'estimated_clv': estimated_clv,
        'churn': churn
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/telecom_churn.csv', index=False)
    print(f"Generated {n_records} records with churn rate: {df['churn'].mean():.2%}")

if __name__ == '__main__':
    generate_enterprise_churn_data()