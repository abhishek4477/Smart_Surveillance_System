import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, source: Union[int, str] = 0, use_gpu: bool = True):
        """
        Initialize camera capture.
        
        Args:
            source: Camera index (0 for default webcam) or IP camera URL
            use_gpu: Whether to use GPU acceleration
        """
        self.source = source
        self.cap = None
        self.is_running = False
        self.use_gpu = use_gpu
        
        # Check if CUDA is available
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("CUDA is available, using GPU acceleration")
            self.gpu_available = True
        else:
            logger.info("CUDA is not available, falling back to CPU")
            self.gpu_available = False
        
    def start(self) -> bool:
        """Start camera capture."""
        try:
            # Set backend to FFMPEG for better RTSP support
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)
                
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                return False
                
            # Set buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set frame size to reduce processing load
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            logger.info(f"Camera started successfully from source: {self.source}")
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if not self.is_running or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return False, None
        
        # Convert frame to GPU if available
        if self.gpu_available:
            frame = cv2.cuda_GpuMat(frame)
        
        return True, frame
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release() 