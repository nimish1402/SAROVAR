import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Split features and target
X = df.drop(columns=["label"])  # Features: N, P, K, temp, humidity, pH, rainfall
y = df["label"]  # Target: Crop name

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with scaling and model
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Feature scaling
    ("model", RandomForestClassifier(random_state=42))  # ML model
])

# Hyperparameter tuning
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)
search.fit(X_train, y_train)

# Evaluate the best model
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized Model Accuracy: {accuracy * 100:.2f}%")

# Save the best model
joblib.dump(best_model, "model/crop_model.pkl")
print("Enhanced model saved successfully!")
