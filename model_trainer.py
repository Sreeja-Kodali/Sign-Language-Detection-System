import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, os

DATA_PATH = "data/gesture_data.csv"
MODEL_PATH = "model/sign_model.pkl"

print("📂 Loading dataset...")

# ✅ Check if dataset exists before proceeding
if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found! Please run the feature extractor first or check your data folder.")
    exit()

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)

# ✅ Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# ✅ Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate model accuracy
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with accuracy: {acc:.2f}")

# ✅ Save trained model
os.makedirs("model", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"💾 Model saved to {MODEL_PATH}")
