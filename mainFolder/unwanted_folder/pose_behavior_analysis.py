import cv2
import numpy as np
import logging
import torch
from ultralytics import YOLO
import time
from typing import List, Dict, Tuple
import math
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseBehaviorAnalyzer:
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize pose-based behavior analyzer using YOLOv8 pose estimation.
        
        Args:
            min_confidence: Minimum confidence threshold for pose detection
        """
        self.min_confidence = min_confidence
        
        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logger.info("CUDA is available, using GPU acceleration")
            torch.cuda.set_device(0)
        else:
            logger.info("CUDA is not available, falling back to CPU")
        
        # Load YOLO pose estimation model
        self.model = YOLO('yolov8n-pose.pt')
        logger.info("Pose behavior analyzer initialized")
        
        # Keypoint indices for different body parts
        self.keypoint_indices = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        
        # Skeleton connections for visualization with angle points
        self.skeleton = [
            # Face
            (0, 1, None), (1, 2, None), (2, 3, None), (3, 1, None),  # Nose to eyes to ears
            # Torso
            (5, 6, None), (5, 11, None), (6, 12, None), (11, 12, None),  # Shoulders and hips
            # Left arm
            (5, 7, 9),  # Left shoulder to elbow to wrist
            # Right arm
            (6, 8, 10),  # Right shoulder to elbow to wrist
            # Left leg
            (11, 13, 15),  # Left hip to knee to ankle
            # Right leg
            (12, 14, 16)   # Right hip to knee to ankle
        ]
        
        # Body part names for labeling
        self.body_parts = {
            'left_arm': (5, 7, 9),
            'right_arm': (6, 8, 10),
            'left_leg': (11, 13, 15),
            'right_leg': (12, 14, 16)
        }
        
        # Motion history for each person
        self.motion_history = {}
        
        # Thresholds for action recognition
        self.thresholds = {
            'sitting_hip_knee_distance': 0.2,  # Normalized distance
            'sitting_torso_angle': 15,  # Degrees from vertical
            'waving_arm_height': 0.1,  # Normalized height above shoulder
            'waving_motion_threshold': 0.05,  # Normalized motion
            'clapping_wrist_distance': 0.15,  # Normalized distance
            'bending_torso_angle': 45,  # Degrees from vertical
            'jumping_vertical_threshold': 0.1,  # Normalized height change
            'falling_rotation_threshold': 30,  # Degrees
            'pointing_arm_angle': 160,  # Degrees
            'writing_hand_height': 0.2,  # Normalized height
            'typing_hand_height': 0.3,  # Normalized height
            'reading_head_angle': 20,  # Degrees
            'phone_hand_height': 0.4  # Normalized height
        }
    
    def get_keypoint(self, keypoints: np.ndarray, name: str) -> Tuple[float, float]:
        """Get coordinates of a specific keypoint."""
        try:
            idx = self.keypoint_indices[name]
            if idx * 3 + 1 < len(keypoints):
                return keypoints[idx * 3], keypoints[idx * 3 + 1]
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Error getting keypoint {name}: {str(e)}")
            return 0.0, 0.0
    
    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        try:
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except Exception as e:
            logger.warning(f"Error calculating angle: {str(e)}")
            return 0.0
    
    def calculate_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        try:
            return np.linalg.norm(np.array(a) - np.array(b))
        except Exception as e:
            logger.warning(f"Error calculating distance: {str(e)}")
            return 0.0
    
    def normalize_distance(self, distance: float, reference: float) -> float:
        """Normalize distance using a reference value."""
        try:
            return distance / reference if reference > 0 else 0
        except Exception as e:
            logger.warning(f"Error normalizing distance: {str(e)}")
            return 0
    
    def update_motion_history(self, person_id: int, keypoints: np.ndarray):
        """Update motion history for a person."""
        if person_id not in self.motion_history:
            self.motion_history[person_id] = deque(maxlen=5)  # Store last 5 frames
        
        # Convert keypoints to numpy array if needed
        if hasattr(keypoints, 'cpu'):
            keypoints = keypoints.cpu().numpy()
        if len(keypoints.shape) > 1:
            keypoints = keypoints[0]  # Take first person's keypoints
        
        # Store current keypoints
        self.motion_history[person_id].append(keypoints)
    
    def detect_motion(self, person_id: int) -> Tuple[float, float]:
        """Detect motion magnitude and direction."""
        if person_id not in self.motion_history or len(self.motion_history[person_id]) < 2:
            return 0.0, 0.0
        
        # Calculate motion between last two frames
        current = self.motion_history[person_id][-1]
        previous = self.motion_history[person_id][-2]
        
        # Ensure both arrays have the same shape
        min_length = min(len(current), len(previous))
        current = current[:min_length]
        previous = previous[:min_length]
        
        # Calculate average motion of all keypoints
        motion_x = np.mean([current[i] - previous[i] for i in range(0, min_length, 3)])
        motion_y = np.mean([current[i+1] - previous[i+1] for i in range(0, min_length, 3)])
        
        return motion_x, motion_y
    
    def analyze_pose(self, keypoints: np.ndarray, person_id: int) -> str:
        """
        Analyze pose keypoints to determine the current action.
        
        Args:
            keypoints: Array of keypoint coordinates and confidences
            person_id: ID of the person being analyzed
            
        Returns:
            String describing the detected action
        """
        try:
            # Update motion history
            self.update_motion_history(person_id, keypoints)
            
            # Get keypoint coordinates
            nose = self.get_keypoint(keypoints, 'nose')
            left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
            right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
            left_elbow = self.get_keypoint(keypoints, 'left_elbow')
            right_elbow = self.get_keypoint(keypoints, 'right_elbow')
            left_wrist = self.get_keypoint(keypoints, 'left_wrist')
            right_wrist = self.get_keypoint(keypoints, 'right_wrist')
            left_hip = self.get_keypoint(keypoints, 'left_hip')
            right_hip = self.get_keypoint(keypoints, 'right_hip')
            left_knee = self.get_keypoint(keypoints, 'left_knee')
            right_knee = self.get_keypoint(keypoints, 'right_knee')
            left_ankle = self.get_keypoint(keypoints, 'left_ankle')
            right_ankle = self.get_keypoint(keypoints, 'right_ankle')
            
            # Calculate reference distances
            shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
            body_height = self.calculate_distance(nose, left_ankle)
            
            # Calculate angles
            torso_angle = self.calculate_angle(nose, left_shoulder, left_hip)
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate distances
            hip_knee_distance = self.normalize_distance(
                self.calculate_distance(left_hip, left_knee),
                body_height
            )
            wrist_distance = self.normalize_distance(
                self.calculate_distance(left_wrist, right_wrist),
                shoulder_width
            )
            
            # Calculate hand heights relative to shoulders
            left_hand_height = (left_wrist[1] - left_shoulder[1]) / body_height
            right_hand_height = (right_wrist[1] - right_shoulder[1]) / body_height
            
            # Detect motion
            motion_x, motion_y = self.detect_motion(person_id)
            
            # Analyze actions based on rules
            if hip_knee_distance < self.thresholds['sitting_hip_knee_distance'] and \
               abs(torso_angle - 90) < self.thresholds['sitting_torso_angle']:
                return "Sitting"
            
            elif abs(torso_angle - 90) < 15 and \
                 abs(left_leg_angle - 180) < 30 and \
                 abs(right_leg_angle - 180) < 30:
                return "Standing"
            
            elif (left_hand_height < -self.thresholds['waving_arm_height'] or \
                  right_hand_height < -self.thresholds['waving_arm_height']) and \
                 abs(motion_x) > self.thresholds['waving_motion_threshold']:
                return "Waving"
            
            elif left_hand_height < -self.thresholds['waving_arm_height'] and \
                 right_hand_height < -self.thresholds['waving_arm_height'] and \
                 abs(motion_x) < self.thresholds['waving_motion_threshold']:
                return "Hands up"
            
            elif (left_hand_height < -self.thresholds['waving_arm_height'] or \
                  right_hand_height < -self.thresholds['waving_arm_height']) and \
                 abs(motion_x) < self.thresholds['waving_motion_threshold']:
                return "Raising arm"
            
            elif wrist_distance < self.thresholds['clapping_wrist_distance'] and \
                 abs(motion_x) > self.thresholds['waving_motion_threshold']:
                return "Clapping"
            
            elif torso_angle > self.thresholds['bending_torso_angle']:
                return "Bending down"
            
            elif abs(motion_y) > self.thresholds['jumping_vertical_threshold']:
                return "Jumping"
            
            elif abs(torso_angle - 90) > self.thresholds['falling_rotation_threshold']:
                return "Falling"
            
            elif (left_arm_angle > self.thresholds['pointing_arm_angle'] or \
                  right_arm_angle > self.thresholds['pointing_arm_angle']) and \
                 abs(motion_x) < self.thresholds['waving_motion_threshold']:
                return "Pointing"
            
            elif abs(left_hand_height) < self.thresholds['writing_hand_height'] and \
                 abs(motion_x) > self.thresholds['waving_motion_threshold']:
                return "Writing"
            
            elif abs(left_hand_height) < self.thresholds['typing_hand_height'] and \
                 abs(right_hand_height) < self.thresholds['typing_hand_height']:
                return "Typing"
            
            elif abs(torso_angle - 90) < self.thresholds['reading_head_angle']:
                return "Reading"
            
            elif abs(left_hand_height) < self.thresholds['phone_hand_height'] and \
                 abs(right_hand_height) < self.thresholds['phone_hand_height']:
                return "Using phone"
            
            else:
                return "Unknown"
                
        except Exception as e:
            logger.warning(f"Error in pose analysis: {str(e)}")
            return "Unknown"
    
    def detect_poses(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect poses in the given frame and analyze behaviors.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing pose detection results and behaviors
        """
        try:
            # Run YOLOv8 pose estimation
            results = self.model(frame, conf=self.min_confidence)
            
            poses = []
            for i, result in enumerate(results):
                boxes = result.boxes
                keypoints = result.keypoints
                
                for j, (box, kpts) in enumerate(zip(boxes, keypoints)):
                    try:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Get keypoints
                        kpts = kpts.cpu().numpy()
                        
                        # Analyze pose
                        action = self.analyze_pose(kpts[0], j + 1)
                        
                        poses.append({
                            'person_id': j + 1,
                            'box': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2)
                            },
                            'confidence': float(confidence),
                            'action': action,
                            'keypoints': kpts[0]  # Store keypoints for visualization
                        })
                    except Exception as e:
                        logger.warning(f"Error processing pose {j}: {str(e)}")
                        continue
            
            return poses
            
        except Exception as e:
            logger.error(f"Error in pose detection: {str(e)}")
            return []
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """
        Draw skeleton, angles, and action labels around detected poses.
        
        Args:
            frame: Input image frame
            poses: List of pose detection results
            
        Returns:
            Frame with drawn skeleton, angles, and labels
        """
        for pose in poses:
            try:
                box = pose['box']
                person_id = pose['person_id']
                action = pose['action']
                keypoints = pose['keypoints']
                
                # Draw skeleton and angles
                for i, j, k in self.skeleton:
                    if i < len(keypoints) // 3 and j < len(keypoints) // 3:
                        pt1 = (int(keypoints[i * 3]), int(keypoints[i * 3 + 1]))
                        pt2 = (int(keypoints[j * 3]), int(keypoints[j * 3 + 1]))
                        
                        # Draw connection line
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                        
                        # If this is a three-point connection (for angle calculation)
                        if k is not None and k < len(keypoints) // 3:
                            pt3 = (int(keypoints[k * 3]), int(keypoints[k * 3 + 1]))
                            angle = self.calculate_angle(pt1, pt2, pt3)
                            self.draw_angle(frame, pt1, pt2, pt3, angle)
                
                # Draw keypoints
                for i in range(len(keypoints) // 3):
                    x = int(keypoints[i * 3])
                    y = int(keypoints[i * 3 + 1])
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                
                # Add person ID and action label
                label = f"Person {person_id}: {action}"
                cv2.putText(frame, label, (box['x1'], box['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add body part labels
                for part_name, (i, j, k) in self.body_parts.items():
                    if i < len(keypoints) // 3 and j < len(keypoints) // 3 and k < len(keypoints) // 3:
                        x = int((keypoints[i * 3] + keypoints[j * 3] + keypoints[k * 3]) / 3)
                        y = int((keypoints[i * 3 + 1] + keypoints[j * 3 + 1] + keypoints[k * 3 + 1]) / 3)
                        cv2.putText(frame, part_name, (x, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception as e:
                logger.warning(f"Error drawing pose: {str(e)}")
                continue
        
        return frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Initialize pose analyzer
    analyzer = PoseBehaviorAnalyzer(min_confidence=0.5)
    
    # Create window
    cv2.namedWindow("Pose Behavior Analysis", cv2.WINDOW_NORMAL)
    
    logger.info("Press 'q' to quit")
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    last_pose_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            current_time = time.time()
            
            # Detect and analyze poses every second
            if current_time - last_pose_time >= 1.0:
                poses = analyzer.detect_poses(frame)
                last_pose_time = current_time
                
                # Print actions in terminal
                if poses:
                    logger.info("Current actions:")
                    for pose in poses:
                        logger.info(f"Person {pose['person_id']}: {pose['action']}")
            else:
                # Use last detected poses
                poses = getattr(analyzer, 'last_poses', [])
            
            # Draw poses
            frame = analyzer.draw_poses(frame, poses)
            
            # Store last poses for visualization between detections
            analyzer.last_poses = poses
            
            # Show frame
            cv2.imshow("Pose Behavior Analysis", frame)
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Current FPS: {fps:.2f}")
            
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