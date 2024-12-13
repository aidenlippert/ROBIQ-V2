import cv2
import numpy as np
import mediapipe as mp

class PoseVisualizer:
    def __init__(self):
        self.connections = mp.solutions.pose.POSE_CONNECTIONS
        self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 0, 0), thickness=2)

    def draw_2d_pose(self, frame, landmarks_2d):
        h, w, _ = frame.shape
        landmarks_px = np.array([[int(l[0] * w), int(l[1] * h), l[2]] for l in landmarks_2d])
        
        for connection in self.connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if landmarks_px[start_idx][2] > 0.5 and landmarks_px[end_idx][2] > 0.5:
                pt1 = tuple(map(int, landmarks_px[start_idx][:2]))
                pt2 = tuple(map(int, landmarks_px[end_idx][:2]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
        for landmark in landmarks_px:
            pt = tuple(map(int, landmark[:2]))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, f'{landmark[2]:.2f}', pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame