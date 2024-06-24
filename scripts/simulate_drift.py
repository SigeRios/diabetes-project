import numpy as np
import pandas as pd

def introduce_drift(data, drift_features, drift_amount=0.1, random_seed=42):
    np.random.seed(random_seed)
    drifted_data = data.copy()
    
    for feature in drift_features:
        if feature in data.columns:
            drifted_data[feature] += np.random.normal(loc=0, scale=drift_amount, size=data.shape[0])
    
    return drifted_data

# Example data (replace these with your actual data)
X_test = pd.DataFrame({
    'Glucose': np.random.normal(100, 20, 100),
    'BloodPressure': np.random.normal(70, 10, 100),
    'SkinThickness': np.random.normal(20, 5, 100),
    'Pregnancies': np.random.randint(0, 10, 100)
})
y_test = pd.Series(np.random.randint(0, 2, 100))

X_train = pd.DataFrame({
    'Glucose': np.random.normal(100, 20, 100),
    'BloodPressure': np.random.normal(70, 10, 100),
    'SkinThickness': np.random.normal(20, 5, 100),
    'Pregnancies': np.random.randint(0, 10, 100)
})
y_train = pd.Series(np.random.randint(0, 2, 100))

# Features to introduce drift
features_to_drift = ['Glucose', 'BloodPressure', 'SkinThickness', 'Pregnancies']

# Introduce drift
drifted_data = introduce_drift(X_test, features_to_drift, drift_amount=50)
drifted_data = drifted_data.reset_index(drop=True)

# Combine with outcome data
reference_data = X_train.copy()
reference_data['Outcome'] = y_train.reset_index(drop=True)
drifted_data['Outcome'] = y_test.reset_index(drop=True)

# Save to CSV
drifted_data.to_csv('./data/new_data.csv', index=False)
reference_data.to_csv('./data/reference_data.csv', index=False)
