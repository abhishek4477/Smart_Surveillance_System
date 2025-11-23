import cv2
import numpy as np
import logging
import os
from datetime import datetime
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize face detector using OpenCV's Haar Cascade classifier.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        self.min_confidence = min_confidence
        
        # Check for CUDA availability
        self.use_cuda = False
        try:
            if torch.cuda.is_available() and hasattr(cv2, 'cuda'):
                self.use_cuda = True
                logger.info("CUDA is available, using GPU acceleration")
            else:
                logger.info("CUDA is not available, falling back to CPU")
        except Exception as e:
            logger.warning(f"CUDA check failed: {str(e)}, falling back to CPU")
            self.use_cuda = False
        
        # Load the Haar Cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            logger.error(f"Haar Cascade file not found at: {cascade_path}")
            raise FileNotFoundError("Haar Cascade file not found")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Face detector initialized")
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing face detection results
        """
        try:
            # Convert to grayscale
            if self.use_cuda:
                try:
                    # Convert frame to GPU
                    frame_gpu = cv2.cuda_GpuMat()
                    frame_gpu.upload(frame)
                    
                    # Convert to grayscale on GPU
                    gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
                    gray = gray_gpu.download()
                except Exception as e:
                    logger.warning(f"GPU processing failed: {str(e)}, falling back to CPU")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to our format
            result = []
            for (x, y, w, h) in faces:
                result.append({
                    'facial_area': {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    },
                    'confidence': 1.0  # Haar Cascade doesn't provide confidence scores
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def draw_faces(self, frame: np.ndarray, faces: list) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            frame: Input image frame
            faces: List of face detection results
            
        Returns:
            Frame with drawn bounding boxes
        """
        # Convert GPU frame to CPU if needed
        if isinstance(frame, cv2.cuda_GpuMat):
            frame = frame.download()
        
        for face in faces:
            facial_area = face.get('facial_area', {})
            if facial_area:
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add face count
                cv2.putText(frame, "Face", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Initialize face detector
    detector = SimpleFaceDetector(min_confidence=0.5)
    
    # Create window
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    
    logger.info("Press 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw faces
            frame = detector.draw_faces(frame, faces)
            
            # Show frame
            cv2.imshow("Face Detection", frame)
            
            # Handle key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 