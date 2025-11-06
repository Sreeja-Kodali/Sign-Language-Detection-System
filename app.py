from flask import Flask, render_template, request
import os, cv2, pickle, numpy as np, pandas as pd, time, gc, gzip

# Try importing mediapipe safely
try:
    import mediapipe as mp
    mp_available = True
except Exception as e:
    print("⚠️ Mediapipe not available in this environment:", e)
    mp_available = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# =================== CONFIG ===================
MODEL_PATH = 'model/sign_model.pkl'
MODEL_PATH_GZ = 'model/sign_model.pkl.gz'
DATA_PATH = 'data/gesture_data.csv'
UPLOAD_FOLDER = '/tmp' if os.environ.get("RENDER") else 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


# =================== TRAIN MODEL ===================
@app.route('/train', methods=['POST'])
def train_model():
    if not os.path.exists(DATA_PATH):
        return render_template('index.html', predicted_sign="❌ Dataset not found!")

    df = pd.read_csv(DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return render_template('index.html', predicted_sign=f"✅ Model trained! Accuracy: {acc:.2f}")


# =================== PREDICT FROM VIDEO ===================
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', predicted_sign="⚠️ No file uploaded!")

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', predicted_sign="⚠️ No file selected!")

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    # Load model
    model = None
    if os.path.exists(MODEL_PATH_GZ):
        with gzip.open(MODEL_PATH_GZ, 'rb') as f:
            model = pickle.load(f)
    elif os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        cleanup(video_path)
        return render_template('index.html', predicted_sign="⚠️ Train model first!")

    # Check mediapipe
    if not mp_available:
        cleanup(video_path)
        return render_template('index.html',
                               predicted_sign="⚠️ Mediapipe not supported on Render. Showing dummy output for demo.")

    # Process video
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                features.append([coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)])
    cap.release()

    if not features:
        cleanup(video_path)
        return render_template('index.html', predicted_sign="❌ No hands detected!")

    avg_features = [sum(col)/len(col) for col in zip(*features)]
    pred = model.predict([avg_features])[0]

    # Label mapping
    label_map = {}
    if os.path.exists("dataset/wlasl_class_list.txt"):
        with open("dataset/wlasl_class_list.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label_map[int(parts[0])] = " ".join(parts[1:])

    label = label_map.get(int(pred), f"Class {pred}")

    cleanup(video_path)
    return render_template('index.html', predicted_sign=f"🧠 Predicted Sign: {label}")


# =================== CLEANUP FUNCTION ===================
def cleanup(path=None):
    """Clean up temporary files, memory, and OpenCV resources."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    gc.collect()
    time.sleep(0.2)

    if path and os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


# =================== RUN APP ===================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
