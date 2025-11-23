import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import time
import torch
import os
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, min_confidence: float = 0.6, process_every_n_frames: int = 1):
        """
        Initialize face detector using RetinaFace with GPU acceleration.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
            process_every_n_frames: Process every nth frame to reduce computation
        """
        self.min_confidence = min_confidence
        self.process_every_n_frames = process_every_n_frames
        self.frame_count = 0
        self.last_faces = []
        self.last_processed_time = 0
        self.processing_interval = 0.05  # Process at most every 50ms
        
        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logger.info("CUDA is available, using GPU acceleration")
            torch.cuda.set_device(0)
        else:
            logger.info("CUDA is not available, falling back to CPU")
        
        # Initialize RetinaFace
        try:
            model_path = os.path.join('models', 'models', 'retinaface_r50_v1')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Please run download_models.py first.")
            
            self.face_analyzer = FaceAnalysis(
                name='retinaface_r50_v1',
                root='models',
                providers=['CUDAExecutionProvider' if self.use_cuda else 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.use_cuda else -1)
            logger.info("Face detector initialized with RetinaFace")
            
        except Exception as e:
            logger.error(f"Error initializing RetinaFace: {str(e)}")
            logger.info("Falling back to OpenCV face detection")
            self.use_retinaface = False
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.use_retinaface = True
    
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Detect faces in the given frame using RetinaFace or OpenCV.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing face detection results
        """
        current_time = time.time()
        
        # Skip processing if not enough time has passed
        if current_time - self.last_processed_time < self.processing_interval:
            return self.last_faces
        
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_faces
        
        try:
            if self.use_retinaface:
                # Detect faces using RetinaFace
                faces = self.face_analyzer.get(frame)
                
                # Convert to our format
                result = []
                for face in faces:
                    if face.det_score >= self.min_confidence:
                        bbox = face.bbox.astype(int)
                        result.append({
                            'facial_area': {
                                'x': bbox[0],
                                'y': bbox[1],
                                'w': bbox[2] - bbox[0],
                                'h': bbox[3] - bbox[1]
                            },
                            'confidence': float(face.det_score),
                            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                            'face_type': 'frontal' if face.det_score > 0.8 else 'profile'
                        })
            else:
                # Fallback to OpenCV face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                result = []
                for (x, y, w, h) in faces:
                    confidence = min(1.0, (w * h) / (frame.shape[0] * frame.shape[1]) * 10)
                    if confidence >= self.min_confidence:
                        result.append({
                            'facial_area': {
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h
                            },
                            'confidence': confidence,
                            'face_type': 'frontal'
                        })
            
            logger.info(f"Detected {len(result)} faces")
            self.last_faces = result
            self.last_processed_time = current_time
            return result
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def draw_faces(self, frame: np.ndarray, faces: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes and visualization cues around detected faces.
        
        Args:
            frame: Input image frame
            faces: List of face detection results
            
        Returns:
            Frame with drawn bounding boxes and cues
        """
        # Draw detection status
        status_text = "Face Detection: Active"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for face in faces:
            facial_area = face.get('facial_area', {})
            if facial_area:
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                face_type = face.get('face_type', 'unknown')
                
                # Draw rectangle around face with different colors for different types
                color = (0, 255, 0) if face_type == 'frontal' else (0, 165, 255)  # Green for frontal, Orange for profile
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw face type label
                cv2.putText(frame, face_type, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add confidence score
                confidence = face.get('confidence', None)
                if confidence:
                    conf_text = f"Conf: {confidence:.2f}"
                    cv2.putText(frame, conf_text, (x, y + h + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw facial landmarks if available
                landmarks = face.get('landmarks')
                if landmarks:
                    for point in landmarks:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)
        
        return frame 