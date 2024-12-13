import mediapipe as mp
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate_pose(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        landmarks_3d = []
        landmarks_2d = []
        
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks_2d.append([landmark.x, landmark.y, landmark.visibility])
            
            for landmark in results.pose_world_landmarks.landmark:
                landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                
        return np.array(landmarks_2d), np.array(landmarks_3d)