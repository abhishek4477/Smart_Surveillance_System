import cv2
import numpy as np
import time
from datetime import datetime
import json
import os

class CrowdManagementSystem:
    def __init__(self):
        self.hog = self._initialize_detector()
        self.people_count = 0
        self.last_update_time = time.time()
        self.update_interval = 5  # Update Omnisight every 5 seconds
        self.data_file = "omnisight_crowd_data.json"

    def _initialize_detector(self):
        # Initialize HOG descriptor for person detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog

    def draw_detection_boxes(self, frame, boxes, weights):
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        
        for (xA, yA, xB, yB) in boxes:
            # Draw detection box
            cv2.rectangle(frame, (int(xA), int(yA)), (int(xB), int(yB)),
                         (0, 255, 0), 2)
        
        # Draw total count and status
        cv2.putText(frame, f'People Count: {len(boxes)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Omnisight Integration Active',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def process_frame(self, frame):
        # Resize frame for faster detection
        frame = cv2.resize(frame, (640, 480))
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(frame, 
                                                 winStride=(8, 8),
                                                 padding=(4, 4),
                                                 scale=1.05)
        
        self.people_count = len(boxes)
        frame = self.draw_detection_boxes(frame, boxes, weights)
        
        # Update Omnisight data if interval has passed
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.update_omnisight_data()
            self.last_update_time = current_time
        
        return frame

    def update_omnisight_data(self):
        data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "people_count": self.people_count,
            "location": "main_entrance",  # Can be configured based on camera location
            "status": "active",
            "alert_level": self.get_alert_level()
        }
        
        # Save to JSON file for Omnisight integration
        self.save_to_json(data)

    def get_alert_level(self):
        # Define crowd density thresholds
        if self.people_count < 5:
            return "LOW"
        elif self.people_count < 15:
            return "MODERATE"
        else:
            return "HIGH"

    def save_to_json(self, data):
        # Load existing data
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = {"crowd_data": []}
        else:
            existing_data = {"crowd_data": []}

        # Append new data
        existing_data["crowd_data"].append(data)
        
        # Keep only last 1000 entries to manage file size
        if len(existing_data["crowd_data"]) > 1000:
            existing_data["crowd_data"] = existing_data["crowd_data"][-1000:]

        # Save updated data
        with open(self.data_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

def main():
    cms = CrowdManagementSystem()
    cap = cv2.VideoCapture(0)
    
    print("Crowd Management System Started - Integrated with Omnisight")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cms.process_frame(frame)
        cv2.imshow('Crowd Management - Omnisight Integration', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 