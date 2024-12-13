import numpy as np

class AngleCalculator:
    def __init__(self):
        self.joint_connections = {
            'left_elbow': [11, 13, 15],  # shoulder, elbow, wrist
            'right_elbow': [12, 14, 16],
            'left_shoulder': [13, 11, 23],  # elbow, shoulder, hip
            'right_shoulder': [14, 12, 24],
            'left_hip': [11, 23, 25],  # shoulder, hip, knee
            'right_hip': [12, 24, 26],
            'left_knee': [23, 25, 27],  # hip, knee, ankle
            'right_knee': [24, 26, 28]
        }

    def calculate_angle(self, p1, p2, p3):
        vector1 = p1 - p2
        vector2 = p3 - p2
        
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def get_all_angles(self, landmarks_3d, landmarks_2d):
        angles = {}
        for joint_name, [p1_idx, p2_idx, p3_idx] in self.joint_connections.items():
            if landmarks_2d[p1_idx][2] > 0.5 and landmarks_2d[p2_idx][2] > 0.5 and landmarks_2d[p3_idx][2] > 0.5:
                p1 = landmarks_3d[p1_idx]
                p2 = landmarks_3d[p2_idx]
                p3 = landmarks_3d[p3_idx]
                angles[joint_name] = self.calculate_angle(p1, p2, p3)
        return angles