from flask import Flask, render_template, request
import os
import cv2
import pickle
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Automatically handle uploads folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'model/sign_model.pkl'

# Load model if exists
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', result="⚠️ No video uploaded")

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', result="⚠️ No file selected")

    # Save uploaded file temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if model is None:
        return render_template('index.html', result="⚠️ Model not found! Train the model first.")

    # Initialize MediaPipe for feature extraction
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    cap = cv2.VideoCapture(filepath)
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
        return render_template('index.html', result="❌ No hands detected in the video")

    # Average features for prediction
    avg_features = np.mean(features, axis=0).reshape(1, -1)

    try:
        prediction = model.predict(avg_features)[0]
        result = f"Predicted Sign: {prediction}"
    except Exception as e:
        result = f"⚠️ Error during prediction: {e}"

    # Optionally delete file after prediction to keep folder empty
    if os.path.exists(filepath):
        os.remove(filepath)

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
