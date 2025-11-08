import cv2
import numpy as np
import serial
import time
import argparse
import os
import glob

# Function to automatically find Arduino serial port
def find_serial_port(explicit_port: str | None = None) -> str | None:
    if explicit_port:
        return explicit_port
    candidates = []
    # Common macOS patterns
    candidates.extend(glob.glob('/dev/tty.usbmodem*'))
    candidates.extend(glob.glob('/dev/tty.usbserial*'))
    # Linux patterns
    candidates.extend(glob.glob('/dev/ttyACM*'))
    candidates.extend(glob.glob('/dev/ttyUSB*'))
    if not candidates:
        return None
    return candidates[0]

# Function to resolve model file paths
def resolve_model_paths(models_dir: str, prototxt: str | None, caffemodel: str | None):
    default_prototxt = os.path.join(models_dir, 'deploy.prototxt')
    default_model = os.path.join(models_dir, 'mobilenet_iter_73000.caffemodel')
    proto_path = prototxt or default_prototxt
    model_path = caffemodel or default_model
    if not os.path.isfile(proto_path):
        raise FileNotFoundError(f"Missing prototxt: {proto_path}. Run Python/download_models.py")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing caffemodel: {model_path}. Run Python/download_models.py")
    return proto_path, model_path

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Integrated car+headlight detection to Arduino')
    parser.add_argument('--models-dir', default=os.path.join(os.path.dirname(__file__), 'models'))
    parser.add_argument('--prototxt', default=None)
    parser.add_argument('--caffemodel', default=None)
    parser.add_argument('--video', default=None, help='Path to video file instead of webcam')
    parser.add_argument('--confidence', type=float, default=0.4)
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--baud', type=int, default=9600)
    return parser.parse_args()

# Parse arguments and setup paths
args = parse_args()
proto_path, model_path = resolve_model_paths(args.models_dir, args.prototxt, args.caffemodel)
PORT = find_serial_port(args.port)
BAUD_RATE = args.baud

# Load MobileNet SSD model for vehicle detection
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
VEHICLE_CLASSES = ["car", "bus", "truck", "motorbike"]

# Initialize video capture
cap = cv2.VideoCapture(0 if args.video is None else args.video)

try:
    arduino = None
    if PORT is not None:
        # Connect to Arduino via serial if available
        arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        print(f"Connected to Arduino on {PORT}. Starting integrated demo... Press ESC to exit.")
    else:
        print("No Arduino port found. Running in video-only mode. Use --port to specify explicitly.")

    last_signal = b'0'  # Start with no detection signal

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Vehicle detection using MobileNet SSD
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        vehicle_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.confidence:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] in VEHICLE_CLASSES:
                    vehicle_detected = True
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"{CLASSES[idx]}: {confidence:.2f}"
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Headlight detection using brightness thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        headlight_detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) > 4 and area / (peri ** 2) > 0.05:
                    headlight_detected = True
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Headlight", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Decision logic: dim high beams if both vehicle and headlights detected
        if vehicle_detected and headlight_detected:
            signal = b'1'  # Dim high beams (oncoming traffic detected)
        else:
            signal = b'0'  # Full high beams (no oncoming traffic)

        # Send signal to Arduino only if it changed
        if signal != last_signal:
            if arduino is not None:
                arduino.write(signal)
                print(f"Sent '{signal.decode()}' to Arduino (Detection: {vehicle_detected}, Headlights: {headlight_detected})")
            last_signal = signal

        # Display the processed frame
        cv2.imshow("Integrated Detection Demo", frame)
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

except serial.SerialException as e:
    print(f"Error: Serial issue on {PORT}. {e}")
finally:
    if 'arduino' in locals():
        arduino.write(b'0')  # Turn off on exit
        arduino.close()
        print("Serial connection closed.")
    cap.release()
    cv2.destroyAllWindows()
    print("Integrated demo ended.")