import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import shap
import matplotlib.pyplot as plt

def run_stage_c():
    print("--- Stage C: MLShark Anomaly Detection ---")
    
    # 1. Load Data (Using the same headers from Person 1) 
    try:
        df = pd.read_csv('KDDTrain_with_headers.csv') 
    except FileNotFoundError:
        print("Error: KDDTrain_with_headers.csv not found.")
        return

    # 2. Data Preparation (Aligning with Stage A)
    # Step A: Filter for ONLY 'normal' traffic to train the "Safety Net"
    train_normal = df[df['label'] == 'normal'].copy()
    
    cols_to_drop = ['label', 'difficulty'] # Use 'difficulty' instead of 'difficulty_level'
    train_normal = train_normal.drop(columns=cols_to_drop, errors='ignore')
    # Only keep columns that actually exist in the dataframe
    existing_cols_to_drop = [c for c in cols_to_drop if c in train_normal.columns]
    
    X_train_normal = train_normal.drop(existing_cols_to_drop, axis=1)
    
    # Step C: Select only numeric features (Isolation Forest requirement)
    X_train_normal = X_train_normal.select_dtypes(include=[np.number])
    
    print(f"Features used for training: {list(X_train_normal.columns)}")


    # 3. Train Isolation Forest 
    # contamination=0.05 assumes 5% of training data might be 'noise'
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X_train_normal)
    print("Isolation Forest trained on normal traffic baseline.")

    # 4. Save Model for Person 5 (Integration) 
    joblib.dump(iso_forest, 'stage_c_isolation_forest.pkl')
    print("Model saved as stage_c_isolation_forest.pkl")

    # 5. Explainability with SHAP 
    # We use a small subset for the explainer to save time
    explainer = shap.TreeExplainer(iso_forest)
    
    # Test on a small sample of the normal data
    test_sample = X_train_normal.sample(5, random_state=42)
    shap_values = explainer.shap_values(test_sample)

    # Generate a summary plot for your report (Requirement: Metrics for reporting) [cite: 2, 3]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_sample, show=False)
    plt.title("Stage C: Feature Importance for Anomaly Detection")
    plt.savefig('stage_c_shap_summary.png')
    print("SHAP analysis complete. Summary plot saved as stage_c_shap_summary.png")

if __name__ == "__main__":
    run_stage_c()
