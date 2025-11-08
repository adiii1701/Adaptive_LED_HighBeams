import cv2
import numpy as np
import argparse
from collections import deque
import os
import time
import math

class NightVehicleDetector:
    def __init__(self):
        # Initialize MobileNet SSD model for vehicle detection
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.prototxt = os.path.join(self.models_dir, "deploy.prototxt")
        self.caffemodel = os.path.join(self.models_dir, "mobilenet_iter_73000.caffemodel")
        
        if not os.path.isfile(self.prototxt) or not os.path.isfile(self.caffemodel):
            raise FileNotFoundError(f"Model files not found in {self.models_dir}")
        
        self.mobilenet_net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        self.mobilenet_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                                 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                 "sofa", "train", "tvmonitor"]
        # Limit to labels this MobileNet-SSD actually predicts
        self.vehicle_classes_mobilenet = {"car", "bus", "motorbike", "bicycle"}
        
        # Vehicle tracking parameters for stable detection across frames
        self.vehicle_tracks = []
        self.track_iou_thresh = 0.3
        self.persist_for_frames = 8
        self.fade_out_frames = 6
        
        # Store headlight detection history for temporal smoothing
        self.vehicle_headlight_history = {}
        
    def iou(self, boxA, boxB):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
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
    
    def detect_vehicles_mobilenet(self, frame, conf_threshold=0.3, min_box_frac=0.001, max_box_frac=0.40):
        """MobileNet SSD vehicle detection with tracking"""
        # Prepare frame for MobileNet inference
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.mobilenet_net.setInput(blob)
        detections = self.mobilenet_net.forward()
        
        # Process detections and filter for vehicles
        current_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                idx = int(detections[0, 0, i, 1])
                if self.mobilenet_classes[idx] in self.vehicle_classes_mobilenet:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Filter by bounding box area to remove noise and oversized detections
                    bw = max(0, endX - startX)
                    bh = max(0, endY - startY)
                    box_frac = (bw * bh) / float(w * h) if w * h > 0 else 0.0
                    if box_frac < min_box_frac or box_frac > max_box_frac:
                        continue
                    
                    current_boxes.append((startX, startY, endX, endY, self.mobilenet_classes[idx], confidence))
        
        # Match current detections with existing tracks for stable tracking
        updated_tracks = []
        used = set()
        for t in self.vehicle_tracks:
            best_j = -1
            best_iou = 0.0
            for j, cb in enumerate(current_boxes):
                if j in used:
                    continue
                iou_val = self.iou(t['box'], cb[:4])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_iou >= self.track_iou_thresh:
                bx = current_boxes[best_j]
                t['box'] = bx[:4]
                t['class'] = bx[4]
                t['confidence'] = bx[5]
                t['ttl'] = self.persist_for_frames
                used.add(best_j)
            else:
                t['ttl'] -= 1
            if t['ttl'] > -self.fade_out_frames:
                updated_tracks.append(t)
        
        # Create new tracks for unmatched detections
        for j, cb in enumerate(current_boxes):
            if j in used:
                continue
            updated_tracks.append({
                'box': cb[:4], 
                'ttl': self.persist_for_frames, 
                'class': cb[4], 
                'confidence': cb[5]
            })
        
        self.vehicle_tracks = updated_tracks
        
        # Convert active tracks to vehicle format for output
        vehicles = []
        for t in self.vehicle_tracks:
            if t['ttl'] <= 0:
                continue
            x1, y1, x2, y2 = t['box']
            vehicles.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'class': t['class'],
                'confidence': t['confidence'],
                'method': 'MobileNet'
            })
        
        return vehicles
    
    def detect_headlights_in_vehicle(self, frame, vehicle_bbox, vehicle_id, threshold=200, min_area=30):
        """Detect headlights ONLY within a detected vehicle bounding box"""
        vx, vy, vw, vh = vehicle_bbox
        
        # Extract vehicle region with small padding to capture headlights at edges
        padding = 5
        x1 = max(0, vx - padding)
        y1 = max(0, vy - padding)
        x2 = min(frame.shape[1], vx + vw + padding)
        y2 = min(frame.shape[0], vy + vh + padding)
        
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return []
        
        # Image enhancement for better headlight detection in low light
        gamma = 1.4
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
        enhanced = cv2.LUT(vehicle_roi, table)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Threshold to find bright regions (potential headlights)
        _, thresh_high = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise and connect nearby pixels
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh_high, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        headlights = []
        roi_h, roi_w = vehicle_roi.shape[:2]
        
        # Filter contours to identify actual headlights
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter by area - headlights should be reasonably sized
            max_area = (roi_w * roi_h * 0.15)  # Max 15% of vehicle area
            
            if min_area < area < max_area:
                # Get bounding rectangle
                hx, hy, hw, hh = cv2.boundingRect(cnt)
                
                # Analyze shape characteristics - headlights are typically circular/oval
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circularity = 4.0 * np.pi * (area / (peri * peri))
                    aspect_ratio = hw / float(hh) if hh > 0 else 0
                    
                    # Apply shape filters
                    circularity_ok = 0.3 < circularity < 1.5
                    aspect_ok = 0.3 < aspect_ratio < 3.0
                    
                    if circularity_ok and aspect_ok:
                        # Verify brightness in the region
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [cnt], 255)
                        mean_intensity = cv2.mean(gray, mask=mask)[0]
                        
                        # Only accept bright regions
                        if mean_intensity > 180:
                            # Position filtering - headlights are typically in front part of vehicle
                            relative_y = hy / roi_h
                            if relative_y < 0.8:  # Not in bottom 20% of vehicle
                                # Convert back to original frame coordinates
                                abs_x = hx + x1
                                abs_y = hy + y1
                                headlights.append({
                                    'bbox': (abs_x, abs_y, hw, hh),
                                    'area': area,
                                    'brightness': mean_intensity,
                                    'circularity': circularity,
                                    'aspect_ratio': aspect_ratio
                                })
        
        # Limit to maximum 3 headlights per vehicle to reduce false positives
        if len(headlights) > 3:
            headlights.sort(key=lambda x: (x['brightness'], x['circularity']), reverse=True)
            headlights = headlights[:3]
        
        # Apply temporal smoothing to reduce flickering
        if vehicle_id not in self.vehicle_headlight_history:
            self.vehicle_headlight_history[vehicle_id] = deque(maxlen=3)
        
        self.vehicle_headlight_history[vehicle_id].append(headlights)
        
        # Return smoothed results (require presence in recent frames)
        if len(self.vehicle_headlight_history[vehicle_id]) < 2:
            return headlights
        
        # Count occurrences of each headlight across recent frames
        all_headlights = []
        for frame_headlights in self.vehicle_headlight_history[vehicle_id]:
            all_headlights.extend(frame_headlights)
        
        # Keep headlights that appear consistently across frames
        persistent_headlights = []
        for headlight in headlights:
            count = sum(1 for hl in all_headlights if self.headlight_similar(hl, headlight))
            if count >= 2:
                persistent_headlights.append(headlight)
        
        # Final limit: ensure we don't exceed 3 headlights even after temporal smoothing
        if len(persistent_headlights) > 3:
            persistent_headlights.sort(key=lambda x: (x['brightness'], x['circularity']), reverse=True)
            persistent_headlights = persistent_headlights[:3]
        
        return persistent_headlights
    
    def detect_headlights_fullframe(self, frame, vehicle_id=-1, threshold=200, min_area=40, roi_top_frac=0.3):
        """Fallback: detect headlights in a lower ROI of the full frame when no vehicles are found."""
        h, w = frame.shape[:2]
        y0 = int(h * max(0.0, min(0.9, roi_top_frac)))
        full_roi_bbox = (0, y0, w, h - y0)
        return self.detect_headlights_in_vehicle(frame, full_roi_bbox, vehicle_id, threshold=threshold, min_area=min_area)
    
    def headlight_similar(self, hl1, hl2, threshold=20):
        """Check if two headlights are similar (same position)"""
        x1, y1, w1, h1 = hl1['bbox']
        x2, y2, w2, h2 = hl2['bbox']
        
        # Check if centers are close
        cx1, cy1 = x1 + w1//2, y1 + h1//2
        cx2, cy2 = x2 + w2//2, y2 + h2//2
        
        distance = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return distance < threshold
    
    
    def process_frame(self, frame, mobilenet_conf=0.3, headlight_threshold=200, 
                     min_box_frac=0.001, max_box_frac=0.40):
        """Main processing function that combines vehicle and headlight detection"""
        # Detect vehicles using MobileNet SSD
        mobilenet_vehicles = self.detect_vehicles_mobilenet(frame, mobilenet_conf, min_box_frac, max_box_frac)
        
        # Detect headlights only within the detected vehicle regions
        all_headlights = []
        for i, vehicle in enumerate(mobilenet_vehicles):
            vehicle_headlights = self.detect_headlights_in_vehicle(
                frame, vehicle['bbox'], i, headlight_threshold
            )
            all_headlights.extend(vehicle_headlights)
        
        # Fallback: if no vehicles (or none yielded headlights), run a full-frame ROI pass
        if not mobilenet_vehicles or not all_headlights:
            fallback_hl = self.detect_headlights_fullframe(frame, vehicle_id=-1, threshold=headlight_threshold)
            all_headlights.extend(fallback_hl)
        
        return {
            'mobilenet_vehicles': mobilenet_vehicles,
            'headlights': all_headlights
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vehicle and headlight detection (MobileNet only)')
    parser.add_argument('--video', default=None, help='Path to video file')
    parser.add_argument('--mobilenet-conf', type=float, default=0.3, help='MobileNet confidence threshold')
    parser.add_argument('--headlight-threshold', type=int, default=200, help='Headlight brightness threshold')
    parser.add_argument('--min-box-frac', type=float, default=0.001, help='Min vehicle box area fraction')
    parser.add_argument('--max-box-frac', type=float, default=0.40, help='Max vehicle box area fraction')
    parser.add_argument('--fps-sync', action='store_true', help='Sync playback to video FPS')
    args = parser.parse_args()
    
    # Initialize the detector with MobileNet model
    try:
        detector = NightVehicleDetector()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Setup video capture (webcam or video file)
    SOURCE_MODE = 1
    VIDEO_PATH = "/Users/adityagadagandla/Desktop/demo/vid.mp4"
    
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0 if SOURCE_MODE == 0 else VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting vehicle and headlight detection...")
    print("Green: MobileNet vehicles, Red: Headlights (only within vehicles)")
    print("Press ESC to exit, SPACE to pause/resume")
    
    # Setup FPS synchronization for smooth playback
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = max(1, int(1000 / fps)) if args.fps_sync else 1
    print(f"Video FPS: {fps:.1f}, Frame delay: {frame_delay}ms")
    
    paused = False
    
    # Main processing loop
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to capture frame.")
                break
        
        # Process frame to detect vehicles and headlights
        results = detector.process_frame(frame, args.mobilenet_conf, 
                                       args.headlight_threshold, args.min_box_frac, args.max_box_frac)
        
        # Draw detected vehicles in green
        for vehicle in results['mobilenet_vehicles']:
            x, y, w, h = vehicle['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw detected headlights in red
        for headlight in results['headlights']:
            x, y, w, h = headlight['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Show pause status if paused
        if paused:
            cv2.putText(frame, "PAUSED - Press SPACE to resume", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the processed frame
        cv2.imshow("Hybrid Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == 27:  # ESC key - exit
            break
        elif key == 32:  # SPACE key - pause/resume
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection ended.")

if __name__ == "__main__":
    main()