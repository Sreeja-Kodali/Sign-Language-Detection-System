import os
import csv
import cv2
from hand_detector import HandDetector

os.makedirs("data", exist_ok=True)
file_path = "data/gesture_data.csv"

# Create CSV file if not exists
if not os.path.exists(file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]] + ["label"])

detector = HandDetector()
cap = cv2.VideoCapture(0)

print("Press A–Z to record gestures. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.find_hands(frame)
    landmarks = detector.get_landmarks()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    if landmarks and 97 <= key <= 122:  # a-z
        label = chr(key).upper()
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(landmarks + [label])
        print(f"Saved gesture: {label}")

    cv2.imshow("Data Collector", frame)

cap.release()
cv2.destroyAllWindows()
