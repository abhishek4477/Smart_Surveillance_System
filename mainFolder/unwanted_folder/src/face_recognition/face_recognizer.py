import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import pickle
import logging
from datetime import datetime
import time
import torch
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, 
                 db_path: str = "data/face_embeddings",
                 recognition_threshold: float = 0.5,
                 model_name: str = "buffalo_l",
                 cache_size: int = 100):
        """
        Initialize face recognizer using InsightFace.
        
        Args:
            db_path: Path to store face embeddings
            recognition_threshold: Threshold for face recognition similarity
            model_name: Name of the model to use for face recognition
            cache_size: Number of recent face embeddings to cache
        """
        self.db_path = db_path
        self.recognition_threshold = recognition_threshold
        self.model_name = model_name
        self.known_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.cache_size = cache_size
        self.last_processed_time = 0
        self.processing_interval = 0.1  # Process at most every 100ms
        
        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logger.info("CUDA is available, using GPU acceleration")
            torch.cuda.set_device(0)
        else:
            logger.info("CUDA is not available, falling back to CPU")
        
        # Initialize InsightFace
        try:
            model_path = os.path.join('models', 'models', model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Please run download_models.py first.")
            
            self.face_analyzer = FaceAnalysis(
                name=model_name,
                root='models',
                providers=['CUDAExecutionProvider' if self.use_cuda else 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.use_cuda else -1)
            logger.info(f"Face recognizer initialized with {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {str(e)}")
            raise
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Load existing embeddings
        self._load_embeddings()
        logger.info(f"Face recognizer initialized with {len(self.known_embeddings)} known faces")
    
    def _load_embeddings(self):
        """Load face embeddings from database."""
        try:
            embedding_file = os.path.join(self.db_path, "embeddings.pkl")
            if os.path.exists(embedding_file):
                with open(embedding_file, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.known_embeddings)} face embeddings")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
    
    def _save_embeddings(self):
        """Save face embeddings to database."""
        try:
            embedding_file = os.path.join(self.db_path, "embeddings.pkl")
            with open(embedding_file, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            logger.info(f"Saved {len(self.known_embeddings)} face embeddings")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def add_face(self, frame: np.ndarray, name: str) -> bool:
        """
        Add a new face to the database.
        
        Args:
            frame: Input image frame containing the face
            name: Name of the person
            
        Returns:
            bool: True if face was added successfully
        """
        try:
            # Detect and extract face embedding
            faces = self.face_analyzer.get(frame)
            
            if faces and len(faces) > 0:
                # Get the face with highest detection score
                face = max(faces, key=lambda x: x.det_score)
                
                if face.det_score >= 0.5:  # Minimum detection confidence
                    self.known_embeddings[name] = face.embedding
                    self._save_embeddings()
                    logger.info(f"Added face embedding for {name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return False
    
    def recognize_face(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face in the given frame.
        
        Args:
            frame: Input image frame containing the face
            
        Returns:
            Tuple of (name, confidence) if face is recognized, else (None, 0.0)
        """
        current_time = time.time()
        
        # Skip processing if not enough time has passed
        if current_time - self.last_processed_time < self.processing_interval:
            return None, 0.0
        
        try:
            # Generate a cache key based on frame content
            cache_key = str(hash(frame.tobytes()))
            
            # Check cache
            if cache_key in self.embedding_cache:
                embedding, timestamp = self.embedding_cache[cache_key]
                if current_time - timestamp < 1.0:  # Cache valid for 1 second
                    return self._find_best_match(embedding)
            
            # Detect and extract face embedding
            faces = self.face_analyzer.get(frame)
            
            if not faces or len(faces) == 0:
                return None, 0.0
            
            # Get the face with highest detection score
            face = max(faces, key=lambda x: x.det_score)
            
            if face.det_score < 0.5:  # Minimum detection confidence
                return None, 0.0
            
            # Update cache
            self.embedding_cache[cache_key] = (face.embedding, current_time)
            if len(self.embedding_cache) > self.cache_size:
                # Remove oldest entry
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            
            self.last_processed_time = current_time
            return self._find_best_match(face.embedding)
            
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return None, 0.0
    
    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the best matching face in the database."""
        best_match = None
        highest_similarity = -1
        
        for name, known_embedding in self.known_embeddings.items():
            similarity = self._compute_similarity(embedding, known_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = name
        
        if highest_similarity >= self.recognition_threshold:
            logger.info(f"Recognized face as {best_match} with similarity {highest_similarity:.2f}")
            return best_match, highest_similarity
        
        logger.debug(f"No match found (best similarity: {highest_similarity:.2f})")
        return None, highest_similarity
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two face embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)) 