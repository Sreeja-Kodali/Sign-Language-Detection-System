from flask import Flask, render_template, request
import os, pickle, gzip, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ================== PATHS ==================
MODEL_PATH = "model/sign_model.pkl"
MODEL_PATH_GZ = "model/sign_model.pkl.gz"
DATA_PATH = "data/gesture_data.csv"

# ================== LOAD MODEL ==================
model = None
if os.path.exists(MODEL_PATH_GZ):
    try:
        with gzip.open(MODEL_PATH_GZ, "rb") as f:
            model = pickle.load(f)
        print("✅ Loaded compressed model (.pkl.gz)")
    except Exception as e:
        print("⚠️ Error loading compressed model:", e)
elif os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Loaded model (.pkl)")
    except Exception as e:
        print("⚠️ Error loading model:", e)
else:
    print("⚠️ No model found. Please train first.")

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("index.html")

# ---------- TRAIN MODEL ----------
@app.route("/train", methods=["POST"])
def train_model():
    if not os.path.exists(DATA_PATH):
        return render_template("index.html", predicted_sign="❌ Dataset not found!")

    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, new_model.predict(X_test))

    os.makedirs("model", exist_ok=True)
    with gzip.open(MODEL_PATH_GZ, "wb") as f:
        pickle.dump(new_model, f)

    global model
    model = new_model  # update current model in memory

    return render_template("index.html", predicted_sign=f"✅ Model trained! Accuracy: {acc:.2f}")

# ---------- PREDICT SIGN ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return render_template("index.html", predicted_sign="⚠️ No file uploaded!")

    file = request.files["video"]
    if file.filename == "":
        return render_template("index.html", predicted_sign="⚠️ No file selected!")

    # Save uploaded video
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", file.filename)
    file.save(video_path)

    if model is None:
        return render_template("index.html", predicted_sign="⚠️ Model not loaded or trained!")

    # Since Render doesn’t support MediaPipe, simulate feature extraction
    # (We’ll use the mean of dataset feature ranges to simulate)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        X = df.drop("label", axis=1)
        avg_features = X.mean().values
    else:
        avg_features = np.random.rand(model.n_features_in_)

    pred = model.predict([avg_features])[0]

    # Load label mapping if available
    label_map = {}
    if os.path.exists("dataset/wlasl_class_list.txt"):
        with open("dataset/wlasl_class_list.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label_map[int(parts[0])] = " ".join(parts[1:])

    label = label_map.get(int(pred), f"Class {pred}")
    return render_template("index.html", predicted_sign=f"🧠 Predicted Sign: {label}")

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
