import cv2
import json
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 160
CONF_THRESHOLD = 45      # CNN softmax threshold
SMOOTH_FRAMES = 7        # prediction smoothing

# ===============================
# LOAD MODEL & LABELS
# ===============================
model = load_model("student_attendance_resnet50.keras", compile=False)

with open("class_labels.json", "r") as f:
    labels = json.load(f)

# ===============================
# FACE DETECTOR
# ===============================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Prediction buffer
pred_buffer = deque(maxlen=SMOOTH_FRAMES)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not found")
    exit()

print("✅ Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(120, 120)
    )

    display_text = "Align face inside box"
    display_color = (0, 0, 255)

    if len(faces) > 0:
        # Pick the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]

        # -------------------------------
        # ADD PADDING (IMPORTANT)
        # -------------------------------
        pad = int(0.35 * fw)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)

        face = frame[y1:y2, x1:x2]

        # Force square crop
        fh2, fw2, _ = face.shape
        size = min(fh2, fw2)
        face = face[:size, :size]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # -------------------------------
        # PREPROCESS FOR RESNET
        # -------------------------------
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        # -------------------------------
        # PREDICT (SMOOTHED)
        # -------------------------------
        pred = model.predict(face, verbose=0)[0]
        pred_buffer.append(pred)

        avg_pred = np.mean(pred_buffer, axis=0)
        class_id = np.argmax(avg_pred)
        confidence = np.max(avg_pred) * 100
        student_id = labels[class_id]

        if confidence > CONF_THRESHOLD:
            display_text = f"{student_id} ({confidence:.1f}%)"
            display_color = (0, 255, 0)
        else:
            display_text = "Unknown"
            display_color = (0, 0, 255)

    # -------------------------------
    # CENTER GUIDE BOX
    # -------------------------------
    box_size = 260
    cx, cy = w // 2, h // 2
    cv2.rectangle(
        frame,
        (cx - box_size // 2, cy - box_size // 2),
        (cx + box_size // 2, cy + box_size // 2),
        (255, 255, 255), 2
    )

    cv2.putText(
        frame,
        display_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        display_color,
        2
    )

    cv2.imshow("ResNet50 Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
