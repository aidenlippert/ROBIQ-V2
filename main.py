from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from src.pose_estimator import PoseEstimator
from src.angle_calculator import AngleCalculator
from src.keypoint_smoother import KeypointSmoother
from src.visualization import PoseVisualizer
import cv2
import sys
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Fitness Tracker")
        
        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.angle_calculator = AngleCalculator()
        self.keypoint_smoother = KeypointSmoother(33)  # MediaPipe has 33 pose landmarks
        self.visualizer = PoseVisualizer()
        
        # Setup UI
        self.setup_ui()
        
        # Setup camera with updated resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000/30))  # Targeting 30 FPS
        
        self.prev_time = time.time()
        
        # Rep counter state
        self.rep_count = 0
        self.squat_state = 'up'  # 'up' or 'down'
        
        # Feedback state to prevent repeated messages
        self.feedback_state = {
            'knee': False,
            'back': False,
            'hip': False
        }
        
        # Set management
        self.set_active = False
        self.target_reps = 0
        self.rep_feedback = []
        self.current_rep_bad = False

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Top Panel: Controls
        control_layout = QHBoxLayout()
        
        # Rep Count Input
        rep_label = QLabel("Set Reps:")
        self.rep_spinbox = QSpinBox()
        self.rep_spinbox.setMinimum(1)
        self.rep_spinbox.setMaximum(100)
        self.rep_spinbox.setValue(10)  # Default value
        
        # Start Button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_set)
        
        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_set)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(rep_label)
        control_layout.addWidget(self.rep_spinbox)
        control_layout.addStretch()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        
        # Middle Panel: Video and Stats
        middle_layout = QHBoxLayout()
        
        # Camera Feed
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)  # Updated size to match camera resolution
        # Alternatively, make it resizable
        # self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        middle_layout.addWidget(self.camera_label)
        
        # Stats Panel
        stats_layout = QVBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        self.angles_label = QLabel("Joint Angles:\n")
        self.rep_label = QLabel("Reps: 0")
        self.feedback_label = QLabel("Feedback: None")
        self.feedback_label.setWordWrap(True)
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.angles_label)
        stats_layout.addWidget(self.rep_label)
        stats_layout.addWidget(self.feedback_label)
        
        middle_layout.addLayout(stats_layout)
        
        # Add Panels to Main Layout
        main_layout.addLayout(control_layout)
        main_layout.addLayout(middle_layout)
        
        # Set Minimum Window Size
        self.setMinimumSize(800, 550)  # Adjust as needed

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
                
        # Process frame
        landmarks_2d, landmarks_3d = self.pose_estimator.estimate_pose(frame)
        
        if len(landmarks_3d) > 0:
            # Smooth 3D keypoints
            smoothed_3d = self.keypoint_smoother.update(landmarks_3d)
            
            # Calculate angles
            angles = self.angle_calculator.get_all_angles(smoothed_3d, landmarks_2d)
            
            # Draw 2D pose
            frame = self.visualizer.draw_2d_pose(frame, landmarks_2d)
            
            # Update angles display
            angles_text = "Joint Angles:\n"
            for joint, angle in angles.items():
                angles_text += f"{joint}: {angle:.1f}°\n"
            self.angles_label.setText(angles_text)
            
            # Rep counting logic
            if 'left_knee' in angles and 'right_knee' in angles:
                left_knee_angle = angles['left_knee']
                right_knee_angle = angles['right_knee']
                knee_angle = (left_knee_angle + right_knee_angle) / 2
                
                if self.squat_state == 'up' and knee_angle < 130:
                    self.squat_state = 'down'
                    current_feedback = self.check_feedback(angles)
                    if self.set_active:
                        self.current_rep_bad = current_feedback
                elif self.squat_state == 'down' and knee_angle > 150:
                    self.squat_state = 'up'
                    if self.set_active:
                        self.rep_count += 1
                        self.rep_label.setText(f"Reps: {self.rep_count}")
                        self.rep_feedback.append(getattr(self, 'current_rep_bad', False))
                        self.feedback_label.setText("Feedback: None")
                        
                        # Check if set is complete
                        if self.rep_count >= self.target_reps:
                            self.set_active = False
                            self.start_button.setEnabled(True)
                            self.stop_button.setEnabled(False)
                            self.show_report()
                
                # Color indicator for squat depth
                if knee_angle < 90:
                    color = (0, 100, 0)  # Dark green
                elif 90 <= knee_angle <= 110:
                    color = (0, 255, 0)  # Bright green
                elif 110 < knee_angle <= 130:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red (out of range)
                
                # Draw color indicator on the frame
                cv2.putText(frame, f"Depth: {knee_angle:.1f}°", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.prev_time = current_time
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))
    
    def check_feedback(self, angles):
        """
        Checks the angles against thresholds and updates the feedback_label.
        Returns True if any bad form is detected in the current rep.
        """
        thresholds = {
            'knee': {
                'min_down': 90,
                'max_down': 130,
                'min_up': 150,
                'max_up': 180
            },
            'back': {
                'min': 80,  # Neutral spine angle
                'max': 100
            },
            'hip': {
                'min': 90,  # Hip angle should not be too forward
                'max': 120
            }
        }
        
        feedback_messages = []
        bad_form = False
        
        # Knees Feedback
        if 'left_knee' in angles and 'right_knee' in angles:
            knee_angle = (angles['left_knee'] + angles['right_knee']) / 2
            if knee_angle < thresholds['knee']['min_down']:
                feedback_messages.append("Don't squat too deep. Keep your knees above 90 degrees.")
                bad_form = True
            elif knee_angle > thresholds['knee']['max_up']:
                feedback_messages.append("Extend your knees fully but don't lock them.")
                bad_form = True
        
        # Back (Spine) Feedback
        if 'left_shoulder' in angles and 'left_hip' in angles:
            back_angle = angles.get('left_shoulder', 90)  # Approximation
            if back_angle < thresholds['back']['min']:
                feedback_messages.append("Straighten your back. Keep your spine neutral.")
                bad_form = True
            elif back_angle > thresholds['back']['max']:
                feedback_messages.append("Don't lean back. Maintain a neutral spine.")
                bad_form = True
        
        # Hips Feedback
        if 'left_hip' in angles and 'left_knee' in angles:
            hip_angle = angles.get('left_hip', 90)  # Approximation
            if hip_angle < thresholds['hip']['min']:
                feedback_messages.append("Keep your hips back. Don't lean forward.")
                bad_form = True
            elif hip_angle > thresholds['hip']['max']:
                feedback_messages.append("Don't tilt your hips too far back.")
                bad_form = True
        
        if feedback_messages:
            self.feedback_label.setText("Feedback: " + " ".join(feedback_messages))
        else:
            self.feedback_label.setText("Feedback: Good form!")
        
        return bad_form
    
    def start_set(self):
        if self.set_active:
            return  # Already active
        
        self.target_reps = self.rep_spinbox.value()
        self.rep_count = 0
        self.rep_feedback = []
        self.set_active = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.rep_label.setText("Reps: 0")
        self.feedback_label.setText("Feedback: None")
    
    def stop_set(self):
        if not self.set_active:
            return
        
        self.set_active = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.show_report()
    
    def show_report(self):
        total_reps = len(self.rep_feedback)
        bad_reps = sum(self.rep_feedback)
        good_reps = total_reps - bad_reps
        
        report = f"Set Completed!\nTotal Reps: {total_reps}\nGood Form: {good_reps}\nBad Form: {bad_reps}"
        
        if bad_reps > 0:
            report += "\n\nDetails:\n"
            for idx, bad in enumerate(self.rep_feedback, 1):
                if bad:
                    report += f"Rep {idx}: Bad form detected.\n"
        
        QMessageBox.information(self, "Set Report", report)
    
    def closeEvent(self, event):
        # Release the camera
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())