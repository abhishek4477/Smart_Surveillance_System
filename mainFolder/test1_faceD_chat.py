import cv2
import numpy as np
import face_recognition
import mediapipe as mp

# --- IP CAMERA STREAM ---
ip_camera_url = "rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101"  # replace with your IP cam URL
cap = cv2.VideoCapture(ip_camera_url)

# --- Mediapipe for FAST face detection ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces with Mediapipe
    results = face_detection.process(rgb_small_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = small_frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract face region
            face = small_frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Resize face to original size for embedding
            face_resized = cv2.resize(face_rgb, (150, 150))  # face_recognition works best ~150x150

            # Get face embedding
            encoding = face_recognition.face_encodings(face_resized)
            if encoding:
                print("Embedding Vector:", encoding[0][:5])  # print first 5 dims for quick check

            # Draw bounding box
            cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', small_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
