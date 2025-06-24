import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class DrugAddictionDetector:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load or create simple ML model
        self.model = self.load_or_create_model()
        
        # Previous frame for motion detection
        self.prev_gray = None
        self.motion_history = []
        
    def load_or_create_model(self):
        """Load existing model or create a simple one for demonstration"""
        model_path = 'drug_detection_model.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            # Create a simple demo model with random data
            # In real application, this would be trained on actual data
            X_demo = np.random.rand(1000, 10)  # 10 features
            y_demo = np.random.randint(0, 2, 1000)  # Binary classification
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_demo, y_demo)
            joblib.dump(model, model_path)
            return model
    
    def extract_features(self, faces, eyes, motion_level):
        """Extract features from face detection and motion analysis"""
        features = []
        
        # Face features
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            features.extend([
                x / 640.0,  # Normalized x position
                y / 480.0,  # Normalized y position
                w / 640.0,  # Normalized width
                h / 480.0   # Normalized height
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Eye features
        eye_count = len(eyes)
        features.append(eye_count / 2.0)  # Normalized eye count (should be ~1 for 2 eyes)
        
        # Motion features
        features.append(motion_level)
        
        # Add some derived features
        if len(faces) > 0:
            face_ratio = w / h if h > 0 else 0
            features.append(face_ratio)
        else:
            features.append(0)
        
        # Stability features (simplified)
        features.extend([0.5, 0.5, 0.5])  # Placeholder stability metrics
        
        return np.array(features).reshape(1, -1)
    
    def detect_drug_signs(self, features):
        """Use ML model to detect potential drug addiction signs"""
        try:
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0].max()
            return prediction, probability
        except:
            return 0, 0.0
    
    def analyze_frame(self, frame):
        """Analyze a single frame for drug addiction signs"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Eye detection within faces
        eyes = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes_in_face = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes_in_face:
                eyes.append((x+ex, y+ey, ew, eh))
        
        # Motion detection
        motion_level = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            motion_level = np.mean(diff) / 255.0
            
            # Keep motion history
            self.motion_history.append(motion_level)
            if len(self.motion_history) > 30:  # Keep last 30 frames
                self.motion_history.pop(0)
        
        self.prev_gray = gray.copy()
        
        # Extract features and make prediction
        features = self.extract_features(faces, eyes, motion_level)
        prediction, confidence = self.detect_drug_signs(features)
        
        results = {
            'face_detected': len(faces) > 0,
            'eye_detected': len(eyes) > 0,
            'drug_signs_detected': bool(prediction),
            'confidence': confidence,
            'faces': faces,
            'eyes': eyes,
            'motion_level': motion_level
        }
        
        return results
