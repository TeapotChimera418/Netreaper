import pandas as pd
import joblib
import numpy as np

# 1. Load your Model
model = joblib.load('stage_c_isolation_forest.pkl')

# 2. Load the Adversarial Samples
X_adv = pd.read_csv('X_adversarial_samples.csv')

# 3.Match the columns to what the Isolation Forest expects
# Isolation Forest was trained on raw numeric columns from the CSV
# We must ensure we aren't using the 'cat__' or 'num__' prefixed names
if any(col.startswith('num__') or col.startswith('cat__') for col in X_adv.columns):
    # Strip the prefixes if they exist
    X_adv.columns = [col.replace('num__', '').replace('cat__', '') for col in X_adv.columns]

# 4. Ensure only the columns seen during FIT are present
# Get the features the model is looking for
model_features = model.feature_names_in_
X_adv_final = X_adv[model_features]

# 5. Predict
predictions = model.predict(X_adv_final)

# 6. Report Results
detected = np.sum(predictions == -1)
print(f"Final Stage C Verification")
print(f"Attacks Caught: {detected} / {len(X_adv_final)}")
print(f"Detection Rate: {(detected / len(X_adv_final)) * 100:.2f}%")
