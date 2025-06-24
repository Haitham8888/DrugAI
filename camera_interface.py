import cv2
import numpy as np
from drug_detector import DrugAddictionDetector
import time

class CameraInterface:
    def __init__(self):
        self.detector = DrugAddictionDetector()
        self.cap = None
        self.is_recording = False
        
    def start_camera(self, camera_index=0):
        """Initialize and start the camera"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def draw_results(self, frame, results):
        """Draw detection results on the frame"""
        height, width = frame.shape[:2]
        
        # Draw status indicators
        status_y = 30
        if results['face_detected']:
            cv2.putText(frame, "Face: Detected", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face: Not Detected", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        status_y += 25
        if results['eye_detected']:
            cv2.putText(frame, "Eyes: Detected", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Eyes: Not Detected", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw drug detection results
        status_y += 40
        if results['drug_signs_detected']:
            cv2.putText(frame, "WARNING: Drug Signs Detected!", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (5, 5), (width-5, height-5), (0, 0, 255), 3)
        else:
            cv2.putText(frame, "No Drug Signs Detected", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw confidence score
        status_y += 25
        confidence_text = f"Confidence: {results['confidence']:.2f}"
        cv2.putText(frame, confidence_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw motion level
        status_y += 20
        motion_text = f"Motion: {results['motion_level']:.3f}"
        cv2.putText(frame, motion_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw face bounding boxes
        if results['faces'] is not None and len(results['faces']) > 0:
            for (x, y, w, h) in results['faces']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw eye bounding boxes
        if results['eyes'] is not None and len(results['eyes']) > 0:
            for (x, y, w, h) in results['eyes']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, "Eye", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return frame
    
    def run_detection(self):
        """Main loop for real-time drug addiction detection"""
        if self.cap is None:
            self.start_camera()
        
        print("Drug Addiction Detection Started")
        print("Press 'q' to quit, 'r' to toggle recording")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Analyze frame for drug signs
            results = self.detector.analyze_frame(frame)
            
            # Draw results on frame
            frame_with_results = self.draw_results(frame, results)
            
            # Add instructions
            cv2.putText(frame_with_results, "Press 'q' to quit", 
                       (10, frame_with_results.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Drug Addiction Detection', frame_with_results)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                print(f"FPS: {fps:.1f}")
                start_time = time.time()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.is_recording = not self.is_recording
                print(f"Recording: {'ON' if self.is_recording else 'OFF'}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera interface closed")
