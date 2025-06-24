from camera_interface import CameraInterface
import sys

def main():
    """Main application entry point"""
    print("=" * 50)
    print("Health Monitoring System - TESTING MODE")
    print("نظام مراقبة الصحة - وضع التجربة")
    print("=" * 50)
    print("This is an experimental system for educational purposes only.")
    print("It uses computer vision and machine learning for health monitoring.")
    print("Currently under testing and development.")
    print()
    
    try:
        # Create camera interface
        camera = CameraInterface()
        
        # Start detection
        camera.run_detection()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and not used by another application")
    finally:
        print("Program ended")

if __name__ == "__main__":
    main()
