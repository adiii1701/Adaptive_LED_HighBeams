from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import argparse

# Parse command line arguments for YOLO detection
parser = argparse.ArgumentParser(description='YOLOv8 vehicle/headlight demo')
parser.add_argument('--video', default="/Users/adityagadagandla/Desktop/demo/vid.mp4", help='Path to video, or omit for webcam')
parser.add_argument('--model', default="/Users/adityagadagandla/Desktop/demo/yolov8n.pt", help='Path to YOLOv8 .pt model')
parser.add_argument('--confidence', type=float, default=0.35, help='Confidence threshold')
parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
args = parser.parse_args()

# Initialize video capture
cap = cv2.VideoCapture(0 if args.video is None else args.video)

# Load YOLO model and get class names
model = YOLO(args.model)
names = model.names if hasattr(model, 'names') else {}

# Filter for vehicle classes only
vehicle_ids = set()
for k, v in names.items():
    if isinstance(v, str) and v in {"car", "truck", "bus", "motorcycle", "motorbike", "bicycle"}:
        vehicle_ids.add(int(k))

# FPS calculation variables
prev_frame_time = 0
new_frame_time = 0

# Main detection loop
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success or img is None:
        break
    
    # Run YOLO inference on the frame
    results = model(img, stream=True, verbose=False, imgsz=args.imgsz, conf=args.confidence)
    
    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Draw corner rectangle around detection
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # Get confidence and class information
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = names.get(cls, str(cls))
            
            # Only display vehicle detections
            if vehicle_ids and cls not in vehicle_ids:
                continue
            cvzone.putTextRect(img, f'{label} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # Display the frame
    cv2.imshow("YOLOv8 Demo", img)
    if cv2.waitKey(1) == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()