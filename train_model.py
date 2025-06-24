import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def create_demo_dataset():
    """Create a demonstration dataset for drug addiction detection"""
    
    # Simulate features that might indicate drug use
    # Features: face_bbox (4), pose_visibility (6)
    n_samples = 2000
    
    # Normal behavior data
    normal_data = []
    for _ in range(n_samples // 2):
        # Normal face detection (stable bounding box)
        face_bbox = np.random.normal([0.3, 0.2, 0.4, 0.6], [0.05, 0.05, 0.05, 0.05])
        # Normal pose visibility (high visibility for key points)
        pose_visibility = np.random.normal([0.9, 0.9, 0.8, 0.8, 0.7, 0.7], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        features = np.concatenate([face_bbox, pose_visibility])
        normal_data.append(features)
    
    # Abnormal behavior data (potential drug signs)
    abnormal_data = []
    for _ in range(n_samples // 2):
        # Unstable face detection (more variation in bounding box)
        face_bbox = np.random.normal([0.3, 0.2, 0.4, 0.6], [0.15, 0.15, 0.1, 0.1])
        # Lower pose visibility (more erratic movement)
        pose_visibility = np.random.normal([0.6, 0.6, 0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        
        features = np.concatenate([face_bbox, pose_visibility])
        abnormal_data.append(features)
    
    # Combine data
    X = np.array(normal_data + abnormal_data)
    y = np.array([0] * len(normal_data) + [1] * len(abnormal_data))  # 0=normal, 1=abnormal
    
    return X, y

def train_model():
    """Train the drug addiction detection model"""
    print("Creating demonstration dataset...")
    X, y = create_demo_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model...")
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Normal', 'Potential Drug Signs']))
    
    # Save model
    model_path = 'drug_detection_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_model()
