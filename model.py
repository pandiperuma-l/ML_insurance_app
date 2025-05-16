# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv('insurance.csv')

X = df.drop('charges', axis=1)
y = df['charges']

# Categorical features
categorical = ['sex', 'smoker', 'region']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical)],
    remainder='passthrough'
)

# Pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'insurance_model.pkl')
