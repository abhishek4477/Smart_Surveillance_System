import cv2
import numpy as np
from src.utils.camera import Camera
from src.face_recognition.face_detector import FaceDetector
from src.face_recognition.face_recognizer import FaceRecognizer
import logging
import os
from datetime import datetime
import time
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_status(frame: np.ndarray, fps: float, face_count: int, recognized_count: int):
    """Draw status information on the frame."""
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw face count
    cv2.putText(frame, f"Faces: {face_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw recognized count
    cv2.putText(frame, f"Recognized: {recognized_count}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw GPU status
    gpu_status = "GPU: Active" if torch.cuda.is_available() else "GPU: Inactive"
    cv2.putText(frame, gpu_status, (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw instructions
    cv2.putText(frame, "Press 'a' to add face, 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA is available, using GPU acceleration")
        torch.cuda.set_device(0)
    else:
        logger.info("CUDA is not available, falling back to CPU")
    
    # Initialize components
    camera = Camera(
        "rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101",
        use_gpu=False  # Use CPU for camera operations
    )
    face_detector = FaceDetector(
        min_confidence=0.5,  # Lower threshold for better detection
        process_every_n_frames=1  # Process every frame for better accuracy
    )
    face_recognizer = FaceRecognizer(
        db_path="mainFolder/data/face_embeddings",
        recognition_threshold=0.5,  # Lower threshold for better recognition
        model_name="buffalo_l",  # Use InsightFace's best model
        cache_size=100
    )
    
    # Create window
    cv2.namedWindow("Face Recognition Test", cv2.WINDOW_NORMAL)
    
    # Start camera
    if not camera.start():
        logger.error("Failed to start camera")
        return
    
    logger.info("Press 'a' to add a new face, 'q' to quit")
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    recognized_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Current FPS: {fps:.2f}")
            
            # Detect faces using RetinaFace
            faces = face_detector.detect_faces(frame)
            
            # Process each detected face
            recognized_count = 0
            for face in faces:
                facial_area = face.get('facial_area', {})
                if facial_area:
                    x = facial_area.get('x', 0)
                    y = facial_area.get('y', 0)
                    w = facial_area.get('w', 0)
                    h = facial_area.get('h', 0)
                    
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Try to recognize the face using InsightFace
                    name, confidence = face_recognizer.recognize_face(face_img)
                    
                    if name:
                        recognized_count += 1
                    
                    # Draw rectangle and name
                    color = (0, 255, 0) if name else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{name if name else 'Unknown'} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw faces with visualization cues
            frame = face_detector.draw_faces(frame, faces)
            
            # Draw status information
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            draw_status(frame, fps, len(faces), recognized_count)
            
            # Show frame
            cv2.imshow("Face Recognition Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a'):
                # If faces are detected, add the first one
                if faces:
                    face = faces[0]
                    facial_area = face.get('facial_area', {})
                    if facial_area:
                        x = facial_area.get('x', 0)
                        y = facial_area.get('y', 0)
                        w = facial_area.get('w', 0)
                        h = facial_area.get('h', 0)
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Get name from user
                        name = input("Enter name for the face: ")
                        if name:
                            if face_recognizer.add_face(face_img, name):
                                logger.info(f"Added face for {name}")
                            else:
                                logger.error("Failed to add face")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        
        # Log final performance
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        logger.info(f"Final FPS: {fps:.2f}")

if __name__ == "__main__":
    main() 