import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('loan_data.csv')

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Define target column
target_column = 'Loan_Status'

# Ensure target column exists
if target_column not in df.columns:
    raise KeyError(f"Column '{target_column}' not found. Available columns: {df.columns}")

# Convert target variable to numeric (Approved = 1, Rejected = 0)
df[target_column] = df[target_column].map({'Approved': 1, 'Rejected': 0})

# Convert categorical features using one-hot encoding
df = pd.get_dummies(df, columns=['Employment_Status'], drop_first=True)

# Save feature names for inference
feature_names = df.drop(columns=[target_column]).columns

# Define features and target
X = df[feature_names]
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model and feature names
with open('model.pkl', 'wb') as file:
    pickle.dump((rf_model, list(feature_names)), file)

# Evaluate model
y_pred = rf_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))