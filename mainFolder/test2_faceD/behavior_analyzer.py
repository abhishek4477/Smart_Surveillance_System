import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import joblib
from face_recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseClassifier:
    def __init__(self, model_path: str = "models/pose_model_latest.pkl"):
        """Initialize pose classifier."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load trained model
        self.model_path = Path(model_path)
        self.model = None
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded trained pose model from {model_path}")
            else:
                logger.warning(f"No trained model found at {model_path}. Using rule-based classification.")
        except Exception as e:
            logger.error(f"Error loading pose model: {str(e)}")
    
    def extract_pose_features(self, landmarks) -> np.ndarray:
        """Extract relevant features from pose landmarks."""
        points = np.zeros((33, 3))
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = [landmark.x, landmark.y, landmark.z]
        
        # Calculate angles and features
        def calculate_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)
        
        # Extract key points
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_elbow = points[13]
        right_elbow = points[14]
        left_wrist = points[15]
        right_wrist = points[16]
        left_hip = points[23]
        right_hip = points[24]
        
        # Calculate features (same as training)
        features = [
            calculate_angle(left_shoulder, left_elbow, left_wrist),  # Left elbow angle
            calculate_angle(right_shoulder, right_elbow, right_wrist),  # Right elbow angle
            calculate_angle(left_shoulder, right_shoulder, 
                          [right_shoulder[0], right_shoulder[1] + 1, right_shoulder[2]]),  # Shoulder angle
            calculate_angle(left_hip, right_hip, 
                          [right_hip[0], right_hip[1] + 1, right_hip[2]]),  # Hip angle
            left_wrist[1] - left_shoulder[1],  # Left hand raise
            right_wrist[1] - right_shoulder[1],  # Right hand raise
            (left_hip[1] + right_hip[1])/2 - (left_shoulder[1] + right_shoulder[1])/2,  # Torso angle
        ]
        
        return np.array(features)
    
    def classify_pose(self, frame) -> tuple:
        """
        Classify pose in frame using trained model or fallback to rule-based.
        
        Returns:
            tuple: (pose_name, landmarks, confidence)
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return "Unknown", None, 0.0
        
        # Extract features
        features = self.extract_pose_features(results.pose_landmarks)
        
        # Use trained model if available
        if self.model is not None:
            try:
                # Get prediction and probability
                pose = self.model.predict([features])[0]
                proba = np.max(self.model.predict_proba([features])[0])
                return pose, results.pose_landmarks, proba
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                # Fallback to rule-based
                pose = self._rule_based_classification(features)
                return pose, results.pose_landmarks, 0.5
        
        # Rule-based classification if no model
        pose = self._rule_based_classification(features)
        return pose, results.pose_landmarks, 0.5
    
    def _rule_based_classification(self, features) -> str:
        """Simple rule-based pose classification as fallback."""
        left_elbow_angle = features[0]
        right_elbow_angle = features[1]
        shoulder_angle = features[2]
        hip_angle = features[3]
        left_hand_raise = features[4]
        right_hand_raise = features[5]
        torso_angle = features[6]
        
        # Sitting detection
        if torso_angle > 0.2:  # Upper body leaning back
            return "Sitting"
        
        # Clapping detection
        if (left_elbow_angle < 60 and right_elbow_angle < 60 and 
            abs(left_hand_raise - right_hand_raise) < 0.1):
            return "Clapping"
        
        # Raised hand detection
        if left_hand_raise < -0.2 or right_hand_raise < -0.2:
            return "Raise Hand"
        
        # Folded hands detection
        if (left_elbow_angle < 90 and right_elbow_angle < 90 and 
            abs(left_hand_raise - right_hand_raise) < 0.1):
            return "Fold Hand"
        
        # Default to standing
        return "Standing"

class BehaviorAnalyzer:
    def __init__(self, 
                 location: str,
                 log_file: str = "behavior_log.csv",
                 detection_interval: float = 1.5):
        """
        Initialize behavior analyzer.
        
        Args:
            location: Location identifier (e.g., "Library", "Classroom")
            log_file: Path to behavior log CSV file
            detection_interval: Minimum time between detections in seconds
        """
        self.location = location
        self.log_file = Path(log_file)
        self.detection_interval = detection_interval
        
        # Initialize components
        self.face_recognizer = FaceRecognizer(
            embeddings_dir="embeddings",
            recognition_threshold=0.6
        )
        self.pose_classifier = PoseClassifier()
        
        # Initialize state
        self.last_detection_time = 0
        self.last_message = ""
        
        # Ensure log file exists
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize behavior log CSV file."""
        if not self.log_file.exists():
            df = pd.DataFrame(columns=['Timestamp', 'Name', 'Pose', 'Location', 'Sentence'])
            df.to_csv(self.log_file, index=False)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a frame for behavior analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (processed_frame, message)
        """
        current_time = datetime.now()
        
        # Check detection interval
        if (current_time - datetime.fromtimestamp(self.last_detection_time)).total_seconds() < self.detection_interval:
            return frame, self.last_message
        
        try:
            # Face recognition
            processed_frame, face_results = self.face_recognizer.process_frame(frame)
            
            # Pose detection
            pose_name, pose_landmarks, confidence = self.pose_classifier.classify_pose(frame)
            
            # Draw pose landmarks if detected
            if pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    processed_frame,
                    pose_landmarks,
                    self.pose_classifier.mp_pose.POSE_CONNECTIONS
                )
            
            # Generate message for each detected person
            messages = []
            for result in face_results:
                name = result["person_name"]
                message = f"{name} is {pose_name.lower()} in the {self.location} with confidence {confidence:.2f}"
                messages.append(message)
                
                # Log behavior
                self._log_behavior(name, pose_name, message)
            
            # Update state
            self.last_detection_time = current_time.timestamp()
            self.last_message = "; ".join(messages) if messages else "No one detected"
            
            return processed_frame, self.last_message
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, "Error processing frame"
    
    def _log_behavior(self, name: str, pose: str, sentence: str):
        """Log behavior to CSV file."""
        try:
            record = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Name': name,
                'Pose': pose,
                'Location': self.location,
                'Sentence': sentence
            }
            
            df = pd.DataFrame([record])
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"Error logging behavior: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass 