import cv2
import numpy as np
import pickle
from hand_detector import HandDetector

# Load the trained model
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the hand detector
detector = HandDetector()
cap = cv2.VideoCapture(0)

# Set window name
cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural feel
    frame = cv2.flip(frame, 1)

    # Detect hands
    frame = detector.find_hands(frame)
    landmarks = detector.get_landmarks()

    # Predict gesture if landmarks found
    if landmarks:
        data = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(data)[0]

        # Draw text box on screen
        cv2.rectangle(frame, (40, 30), (400, 100), (0, 0, 0), -1)  # Black background box
        cv2.putText(frame, f"Detected: {prediction}", (50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        cv2.rectangle(frame, (40, 30), (400, 100), (0, 0, 0), -1)
        cv2.putText(frame, "No Hand Detected", (50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    # Display the webcam feed
    cv2.imshow("Sign Language Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
