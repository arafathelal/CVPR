import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mnist_model.keras")

cap = cv2.VideoCapture(0)  # try 1 if 0 fails

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    roi = gray[h//3:2*h//3, w//3:2*w//3]

    # ---- PREPROCESS (MNIST STYLE) ----
    roi = cv2.GaussianBlur(roi, (7, 7), 0)
    _, roi = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    roi = cv2.resize(roi, (28, 28))

    roi = roi / 255.0
    roi = roi.reshape(1, 28, 28)

    # ---- PREDICT ----
    probs = model.predict(roi, verbose=0)
    digit = np.argmax(probs)
    confidence = float(np.max(probs))  # convert to float

    # ---- DISPLAY LABEL + CONFIDENCE ----
    if confidence > 0.80:
        label = f"Digit: {digit} ({confidence*100:.1f}%)"
    else:
        label = f"Digit: ? ({confidence*100:.1f}%)"

    cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), (0,255,0), 2)
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 3)

    cv2.imshow("MNIST Webcam", frame)
    cv2.imshow("ROI", cv2.resize(roi[0], (200, 200)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
