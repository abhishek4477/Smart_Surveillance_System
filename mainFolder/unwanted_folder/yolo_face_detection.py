import cv2
import numpy as np
import logging
import torch
from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOFaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize face detector using YOLOv8 with face-specific parameters.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        self.min_confidence = min_confidence
        
        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logger.info("CUDA is available, using GPU acceleration")
            # Set device to GPU
            torch.cuda.set_device(0)
        else:
            logger.info("CUDA is not available, falling back to CPU")
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        logger.info("Face detector initialized")
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in the given frame using YOLOv8 with face-specific parameters.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing face detection results
        """
        try:
            # Run YOLOv8 inference with face-specific parameters
            results = self.model(frame, 
                               classes=0,  # Person class
                               conf=self.min_confidence,
                               iou=0.3,  # Lower IOU threshold for better face detection
                               imgsz=640)  # Standard image size
            
            faces = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Calculate box dimensions
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Filter for face-like aspect ratios (typically faces are roughly square)
                    aspect_ratio = width / height
                    if 0.7 <= aspect_ratio <= 1.3:  # Face-like aspect ratio
                        faces.append({
                            'facial_area': {
                                'x': int(x1),
                                'y': int(y1),
                                'w': int(width),
                                'h': int(height)
                            },
                            'confidence': float(confidence)
                        })
            
            return faces
            
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
        for face in faces:
            facial_area = face.get('facial_area', {})
            if facial_area:
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add confidence score
                confidence = face.get('confidence', None)
                if confidence:
                    conf_text = f"Face: {confidence:.2f}"
                    cv2.putText(frame, conf_text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Initialize face detector with higher confidence threshold
    detector = YOLOFaceDetector(min_confidence=0.6)
    
    # Create window
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    
    logger.info("Press 'q' to quit")
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    
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
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Current FPS: {fps:.2f}")
                logger.info(f"Faces detected: {len(faces)}")
            
            # Handle key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Log final performance
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        logger.info(f"Final FPS: {fps:.2f}")

if __name__ == "__main__":
    main() 