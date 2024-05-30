import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/object_tracker.h5')

def predict_direction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    predictions = model.predict(resized)
    return np.argmax(predictions)

directions = ['left', 'right', 'up', 'down']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    direction = predict_direction(frame)
    cv2.putText(frame, directions[direction], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
