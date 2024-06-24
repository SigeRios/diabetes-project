import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
reference_data = pd.read_csv('data/reference_data.csv')
new_data = pd.read_csv('data/new_data.csv')

# Combine reference and new data
df = pd.concat([reference_data, new_data], ignore_index=True)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handle missing values in target variable
df = df.dropna(subset=['Outcome'])
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Define the preprocessing pipeline
numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline to the data
pipeline.fit(X, y)

# Save the trained pipeline
with open('models/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
