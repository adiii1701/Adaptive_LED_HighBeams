import cv2
import os
import math

# Load MobileNet SSD model for vehicle detection
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODELS_DIR, "mobilenet_iter_73000.caffemodel")

if not os.path.isfile(PROTOTXT) or not os.path.isfile(CAFFEMODEL):
    raise FileNotFoundError(
        f"Model files not found in {MODELS_DIR}. Expected 'deploy.prototxt' and 'mobilenet_iter_73000.caffemodel'."
    )

net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

# MobileNet class names
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Filter for road vehicles only
VEHICLE_CLASSES = {"car", "bus", "motorbike", "bicycle","truck", "vehicle"}

# Bounding box area filters to remove noise and oversized detections
MIN_BOX_FRAC = 0.005   # Minimum 0.5% of frame area
MAX_BOX_FRAC = 0.40    # Maximum 40% of frame area

# Calculate Intersection over Union for tracking
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0

# Initialize video capture
VIDEO_PATH = "/Users/adityagadagandla/Desktop/demo/vid.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Vehicle tracking parameters for stable detection
TRACK_IOU_THRESH = 0.3
PERSIST_FOR_FRAMES = 6
FADE_OUT_FRAMES = 6
tracks = []  # List to store tracked vehicles
# Main detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame for MobileNet inference
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Process detections and filter for vehicles
    current_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in VEHICLE_CLASSES:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                # Ensure coordinates are within frame bounds
                startX = max(0, min(startX, w - 1))
                endX = max(0, min(endX, w - 1))
                startY = max(0, min(startY, h - 1))
                endY = max(0, min(endY, h - 1))
                bw = max(0, endX - startX)
                bh = max(0, endY - startY)
                box_frac = (bw * bh) / float(w * h) if w * h > 0 else 0.0
                # Filter by area to remove noise and oversized detections
                if box_frac < MIN_BOX_FRAC or box_frac > MAX_BOX_FRAC:
                    continue
                current_boxes.append((startX, startY, endX, endY, CLASSES[idx], float(confidence)))

    # Update tracks with current detections
    updated_tracks = []
    used = set()
    for t in tracks:
        best_j = -1
        best_iou = 0.0
        for j, cb in enumerate(current_boxes):
            if j in used:
                continue
            iou_val = iou(t['box'], cb[:4])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= TRACK_IOU_THRESH:
            # refresh track
            bx = current_boxes[best_j]
            t['box'] = bx[:4]
            t['label'] = bx[4]
            t['conf'] = bx[5]
            t['ttl'] = PERSIST_FOR_FRAMES
            used.add(best_j)
        else:
            # no match this frame; decay ttl
            t['ttl'] -= 1
        if t['ttl'] > -FADE_OUT_FRAMES:
            updated_tracks.append(t)
    # add unmatched current boxes as new tracks
    for j, cb in enumerate(current_boxes):
        if j in used:
            continue
        updated_tracks.append({'box': cb[:4], 'ttl': PERSIST_FOR_FRAMES, 'label': cb[4], 'conf': cb[5]})
    tracks = updated_tracks

    # Draw tracks that are still alive
    for t in tracks:
        if t['ttl'] <= 0:
            # optional: draw faded box; we skip drawing for speed/stability
            continue
        x1, y1, x2, y2 = t['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # optional label suppressed to reduce flicker

    cv2.imshow("Car Detection Demo", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
