import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv("data/gesture_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as sign_model.pkl")
