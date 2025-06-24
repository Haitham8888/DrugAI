from camera_interface import CameraInterface
import sys

def main():
    """Main application entry point"""
    print("=" * 50)
    print("Drug Addiction Detection System")
    print("=" * 50)
    print("This is a demonstration system for educational purposes only.")
    print("It uses computer vision and machine learning to detect potential signs.")
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
