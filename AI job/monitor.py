import cv2
import time
from datetime import datetime

# Load Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0) https://github.com/TLPRODIGY0079/team-project-1

# Frame rate control
last_logged_time = 0
frame_delay = 1  # seconds between logs to avoid spam

print("Starting webcam face detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Timestamped status logging
    current_time = time.time()
    if current_time - last_logged_time > frame_delay:
        if len(faces) > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] You're working! ✅")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No face detected (idle?) 💤")
        last_logged_time = current_time

    # Show webcam frame
    cv2.imshow("Face Detection - Press 'q' to Quit", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
