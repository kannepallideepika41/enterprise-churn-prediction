import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from xgboost import XGBClassifier

def train_pipeline():
    df = pd.read_csv('data/telecom_churn.csv')
    
    # Feature Engineering
    df['spend_to_tenure_ratio'] = df['monthly_spend'] / (df['tenure_months'] + 1)
    df['risk_score_index'] = (df['support_tickets_30d'] * 1.5) + (df['failed_payments_6m'] * 2.0)
    
    features = [
        'tenure_months', 'monthly_spend', 'total_charges', 
        'contract_type', 'payment_method', 'support_tickets_30d', 
        'usage_drop_ratio', 'failed_payments_6m', 'spend_to_tenure_ratio', 'risk_score_index'
    ]
    target = 'churn'
    
    X = df[features]
    y = df[target]
    
    num_cols = [
        'tenure_months', 'monthly_spend', 'total_charges', 
        'support_tickets_30d', 'usage_drop_ratio', 'failed_payments_6m',
        'spend_to_tenure_ratio', 'risk_score_index'
    ]
    cat_cols = ['contract_type', 'payment_method']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    clf = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    print("Training XGBoost Pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    # Validation
    y_preds = model_pipeline.predict(X_test)
    y_probs = model_pipeline.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    print("\n--- Model Evaluation ---")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_preds))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/churn_xgb_pipeline.joblib')
    print("Saved pipeline to models/churn_xgb_pipeline.joblib")

if __name__ == '__main__':
    train_pipeline()