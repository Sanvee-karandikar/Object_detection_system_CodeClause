from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("models/yolov8n.pt")  # Ensure the model file exists in the correct path

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional, adjust as needed)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run object detection
    results = model(frame)

    # Display results
    annotated_frame = results[0].plot()  # Draw bounding boxes
    cv2.imshow("Object Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()