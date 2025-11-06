from flask import Flask, render_template, request
from rq import Queue
from redis import Redis
import os, cv2, pickle, numpy as np, pandas as pd, gzip, mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc

app = Flask(__name__)

MODEL_PATH = 'model/sign_model.pkl'
MODEL_PATH_GZ = 'model/sign_model.pkl.gz'
DATA_PATH = 'data/gesture_data.csv'

# Connect to Redis (Render automatically gives a Redis URL if you add Redis service)
redis_conn = Redis(host=os.getenv('REDIS_HOST', 'localhost'),
                   port=int(os.getenv('REDIS_PORT', 6379)))
queue = Queue(connection=redis_conn)

@app.route('/')
def home():
    return render_template('index.html')

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

    os.makedirs('model', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return render_template('index.html', predicted_sign=f"✅ Model trained! Accuracy: {acc:.2f}")

# ============== Background processing ==============
def process_video(video_path):
    # Load model
    model = None
    if os.path.exists(MODEL_PATH_GZ):
        with gzip.open(MODEL_PATH_GZ, 'rb') as f:
            model = pickle.load(f)
    elif os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        return "⚠️ Train model first!"

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_skip = 3
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                features.append([coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)])
    cap.release()
    cv2.destroyAllWindows()
    gc.collect()

    if not features:
        return "❌ No hands detected!"

    avg_features = [sum(col)/len(col) for col in zip(*features)]
    pred = model.predict([avg_features])[0]

    # Load labels
    label_map = {}
    if os.path.exists("dataset/wlasl_class_list.txt"):
        with open("dataset/wlasl_class_list.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label_map[int(parts[0])] = " ".join(parts[1:])
    label = label_map.get(int(pred), f"Class {pred}")

    os.remove(video_path)
    return f"🧠 Predicted Sign: {label}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', predicted_sign="⚠️ No file uploaded!")

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', predicted_sign="⚠️ No file selected!")

    os.makedirs('uploads', exist_ok=True)
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)

    job = queue.enqueue(process_video, video_path)
    return render_template('index.html',
                           predicted_sign=f"⏳ Processing... Job ID: {job.id}")

@app.route('/status/<job_id>')
def job_status(job_id):
    from rq.job import Job
    job = Job.fetch(job_id, connection=redis_conn)
    if job.is_finished:
        return render_template('index.html', predicted_sign=job.result)
    elif job.is_failed:
        return render_template('index.html', predicted_sign="❌ Job failed.")
    else:
        return render_template('index.html', predicted_sign="⏳ Still processing...")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
