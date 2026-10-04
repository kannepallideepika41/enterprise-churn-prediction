# Enterprise Customer Churn & Lifetime Value Predictive Engine

An end-to-end predictive machine learning pipeline and interactive intelligence simulator designed to forecast subscription customer churn and quantify revenue recovery opportunities.

---

## Architecture Overview

1. **Synthetic Data Generation Engine**: Generates 70,000+ realistic subscription records modeling complex non-linear churn risk dynamics, payment anomalies, and customer lifetime value (CLV).
2. **Feature Engineering & Preprocessing**:
   - `spend_to_tenure_ratio`: Identifies high-risk early-stage high-spenders.
   - `risk_score_index`: Composite interaction index combining recent support tickets and failed billing attempts.
   - Robust scaling on continuous metrics and one-hot encoding on categorical attributes.
3. **Machine Learning Model**:
   - **XGBoost Classifier** with dynamic class weighting (`scale_pos_weight`) to resolve churn class imbalance.
   - Evaluated via Precision-Recall AUC (PR-AUC) and ROC-AUC metrics.
4. **Interactive Analyst Simulator**: Built with **Streamlit** to provide real-time risk classification, baseline CLV projection, and automated retention playbooks.

---

## Directory Layout

```text
enterprise-churn-prediction/
│
├── data/
│   └── generate_data.py          # Synthetic dataset generator
├── src/
│   └── train.py                  # End-to-end preprocessing & XGBoost training pipeline
├── models/
│   └── churn_xgb_pipeline.joblib # Serialized model pipeline
├── app.py                        # Streamlit analyst UI & risk simulator
├── requirements.txt
└── README.md