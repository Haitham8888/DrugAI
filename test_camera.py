#!/usr/bin/env python3
"""
Simple test script to check camera functionality
"""
import cv2
import sys

def test_camera():
    """Test if camera is working"""
    print("Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera!")
        print("Please check:")
        print("1. Camera is connected")
        print("2. Camera permissions are granted")
        print("3. No other application is using the camera")
        return False
    
    print("Camera opened successfully!")
    
    # Test reading a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera!")
        cap.release()
        return False
    
    print(f"Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("Camera test successful!")
    
    # Show a test frame for 3 seconds
    cv2.imshow('Camera Test - Press any key to close', frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    cap.release()
    return True

if __name__ == "__main__":
    if test_camera():
        print("\n✅ Camera is working! You can now run: python main.py")
    else:
        print("\n❌ Camera test failed!")
        sys.exit(1)
