import cv2  # OpenCV for computer vision operations
import serial  # Serial communication with Arduino
import time  # Time utilities for tracking and delays
import numpy as np  # Numerical operations and array handling
from ultralytics import YOLO  # YOLOv8 model for vehicle detection
import os  # File system operations
import glob  # File pattern matching
import argparse  # Command line argument parsing

# ============ CONFIGURATION ============
ARDUINO_PORT = '/dev/tty.usbserial-A5069RR4'  # Serial port for Arduino communication
BAUD_RATE = 9600  # Serial communication baud rate
VIDEO_FOLDER = '/Users/adityagadagandla/Desktop/demo/videos'  # Default folder to search for videos
VIDEO_EXTENSIONS = ['*.mov', '*.MOV', '*.mp4', '*.MP4']  # Supported video file extensions

YOLO_MODEL = 'yolov8n.pt'  # YOLOv8 nano model file (lightweight, fast)
CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence score for vehicle detections (35%)
VEHICLE_CLASSES = [2, 3, 5, 7]  # YOLO class IDs: car=2, motorcycle=3, bus=5, truck=7

TOTAL_COLUMNS = 16  # Total number of columns for adaptive headlight control (8 per LED matrix)
COOLDOWN_SECONDS = 2.0  # Time in seconds a column stays blocked after last detection
MOTION_THRESHOLD = 15  # Minimum pixel movement to consider a vehicle as moving
HEADLIGHT_THRESHOLD = 200  # Brightness threshold (0-255) for headlight detection
HEADLIGHT_MIN_AREA = 30  # Minimum contour area in pixels to be considered a headlight
HEADLIGHT_FALLBACK_ROI_TOP_FRAC = 0.3  # Fraction from top to ignore when no vehicles detected (uses bottom 70%)

# ============ INITIALIZE ============
print("=" * 70)
print("ADAPTIVE HEADLIGHT - 16 COLUMN PRECISION")
print("=" * 70)

# Parse command line arguments for video file input
parser = argparse.ArgumentParser(description='16-column adaptive headlight control')
parser.add_argument('--video', default=None, help='Path to a single video file to process')
args = parser.parse_args()

# Attempt to connect to Arduino via serial port
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)  # Open serial connection
    time.sleep(2)  # Wait for Arduino to initialize
    print("✓ Arduino connected")
except:
    print("✗ Arduino not found")  # Continue without Arduino if connection fails
    arduino = None

# Function to find all video files in a folder matching supported extensions
def find_videos(folder):
    files = []
    for ext in VIDEO_EXTENSIONS:  # Search for each extension pattern
        files.extend(glob.glob(os.path.join(folder, ext)))  # Add matching files to list
    return sorted(files)  # Return sorted list of video files

# Get video list: use command line argument if provided, otherwise search folder
videos = [args.video] if args.video else find_videos(VIDEO_FOLDER)
print(f"✓ Found {len(videos)} videos")

# Load YOLOv8 model for vehicle detection
model = YOLO(YOLO_MODEL)
print("✓ YOLO loaded")
print("=" * 70)

# ============ TRACKING ============
vehicle_tracker = {}  # Global dictionary to store tracked vehicles by ID

class VehicleTracker:
    """Tracks individual vehicles across frames to determine motion and persistence"""
    def __init__(self, vehicle_id, bbox, side):
        self.id = vehicle_id  # Unique identifier for this vehicle
        self.positions = [bbox]  # History of bounding box positions (x1, y1, x2, y2)
        self.side = side  # Which side of frame: 'left' or 'right'
        self.last_seen = time.time()  # Timestamp of last detection
        
    def update(self, bbox):
        """Add new position to tracking history"""
        self.positions.append(bbox)  # Append current bounding box
        self.last_seen = time.time()  # Update last seen timestamp
        if len(self.positions) > 5:  # Keep only last 5 positions to limit memory
            self.positions.pop(0)  # Remove oldest position
    
    def is_moving(self):
        """Check if vehicle has moved significantly between first and last position"""
        if len(self.positions) < 2:  # Need at least 2 positions to determine movement
            return False
        x1_start, x1_end = self.positions[0][0], self.positions[-1][0]  # X coordinates
        y1_start, y1_end = self.positions[0][1], self.positions[-1][1]  # Y coordinates
        movement = abs(x1_end - x1_start) + abs(y1_end - y1_start)  # Total pixel movement
        return movement > MOTION_THRESHOLD  # True if movement exceeds threshold
    
    def is_expired(self, current_time):
        """Check if vehicle hasn't been seen for more than 1 second"""
        return (current_time - self.last_seen) > 1.0  # Expired if unseen for 1+ seconds

# ============ FUNCTIONS ============
def send_cmd(cmd):
    """Send command string to Arduino via serial if connected"""
    if arduino:  # Only send if Arduino is connected
        try:
            arduino.write((cmd + '\n').encode())  # Encode string to bytes and add newline
            time.sleep(0.05)  # Small delay to ensure command is processed
        except:
            pass  # Silently fail if serial write fails

def is_oncoming(x, y, w, h, fw, fh):
    """Check if vehicle is in the oncoming traffic zone (center region of frame)"""
    cy, cx = y + h//2, x + w//2  # Calculate center point of bounding box
    vp, hp = cy/fh, cx/fw  # Convert to normalized position (0.0 to 1.0)
    if vp < 0.2 or vp > 0.85 or hp < 0.05 or hp > 0.95:  # Filter out vehicles too close to edges
        return False
    if (w*h)/(fw*fh) < 0.003:  # Filter out vehicles smaller than 0.3% of frame area
        return False
    return True  # Vehicle is in valid oncoming traffic zone

def match_vehicle(bbox, side, fw):
    """Match current detection to existing tracked vehicle or create new tracker"""
    global vehicle_tracker
    x1, y1, x2, y2 = bbox  # Extract bounding box coordinates
    cx, cy = (x1+x2)//2, (y1+y2)//2  # Calculate center point
    
    min_dist = float('inf')  # Initialize minimum distance to infinity
    matched_id = None  # ID of best matching vehicle
    
    # Search through all existing tracked vehicles
    for vid, tracker in vehicle_tracker.items():
        if tracker.side != side:  # Only match vehicles on same side of frame
            continue
        last_bbox = tracker.positions[-1]  # Get last known position
        last_cx = (last_bbox[0] + last_bbox[2]) // 2  # Center X of last position
        last_cy = (last_bbox[1] + last_bbox[3]) // 2  # Center Y of last position
        dist = ((cx - last_cx)**2 + (cy - last_cy)**2) ** 0.5  # Euclidean distance
        
        if dist < min_dist and dist < 100:  # If closer than previous match and within 100 pixels
            min_dist = dist  # Update minimum distance
            matched_id = vid  # Update matched vehicle ID
    
    if matched_id:  # If found a match
        vehicle_tracker[matched_id].update(bbox)  # Update existing tracker with new position
        return matched_id
    else:  # No match found, create new tracker
        new_id = len(vehicle_tracker) + 1  # Generate new unique ID
        vehicle_tracker[new_id] = VehicleTracker(new_id, bbox, side)  # Create new tracker
        return new_id

def analyze_vehicles_16col(detections, fw, fh):
    """Analyze vehicles and map their positions to 16 columns for headlight control"""
    global vehicle_tracker
    current_time = time.time()
    
    # Remove expired trackers (vehicles not seen for 1+ seconds)
    expired = [vid for vid, t in vehicle_tracker.items() if t.is_expired(current_time)]
    for vid in expired:
        del vehicle_tracker[vid]  # Clean up old trackers
    
    vehicles = []  # List of valid moving vehicles
    blocked_columns = [False] * TOTAL_COLUMNS  # Boolean array for each column (0-15)
    
    # Process each detected vehicle
    for det in detections:
        x1, y1, x2, y2 = det['bbox']  # Extract bounding box coordinates
        w, h = x2-x1, y2-y1  # Calculate width and height
        
        if is_oncoming(x1, y1, w, h, fw, fh):  # Check if vehicle is in valid zone
            cx = (x1+x2)//2  # Calculate horizontal center
            mid = fw//2  # Frame midpoint
            side = 'left' if cx < mid else 'right'  # Determine which side of frame
            
            vid = match_vehicle((x1, y1, x2, y2), side, fw)  # Match to existing or create new tracker
            tracker = vehicle_tracker[vid]  # Get tracker for this vehicle
            is_moving = tracker.is_moving()  # Check if vehicle is moving
            
            if is_moving:  # Only block columns for moving vehicles
                # Map vehicle center to column number (0-15)
                col = int((cx / fw) * TOTAL_COLUMNS)  # Convert X position to column index
                col = max(0, min(TOTAL_COLUMNS - 1, col))  # Clamp to valid range
                
                # Block the column containing the vehicle
                blocked_columns[col] = True
                if w > fw * 0.10:  # If vehicle is large (wider than 10% of frame)
                    if col > 0:  # Block left adjacent column if not at edge
                        blocked_columns[col - 1] = True
                    if col < TOTAL_COLUMNS - 1:  # Block right adjacent column if not at edge
                        blocked_columns[col + 1] = True
                
                # Store vehicle information for visualization
                vehicles.append({
                    'bbox': (x1, y1, x2, y2),
                    'side': side,
                    'class': det['class_name'],
                    'moving': is_moving,
                    'tracker_id': vid,
                    'column': col
                })
    
    return vehicles, blocked_columns  # Return vehicles list and column blocking array

def _detect_headlights_in_bbox(frame, bbox, threshold=HEADLIGHT_THRESHOLD, min_area=HEADLIGHT_MIN_AREA):
    """Detect headlights within a bounding box region using image processing"""
    vx, vy, vw, vh = bbox  # Extract bounding box coordinates
    padding = 5  # Add padding to capture headlights at edges
    x1 = max(0, vx - padding)  # Left boundary with padding, clamped to frame
    y1 = max(0, vy - padding)  # Top boundary with padding, clamped to frame
    x2 = min(frame.shape[1], vx + vw + padding)  # Right boundary with padding
    y2 = min(frame.shape[0], vy + vh + padding)  # Bottom boundary with padding

    roi = frame[y1:y2, x1:x2]  # Extract region of interest from frame
    if roi.size == 0:  # Return empty if ROI is invalid
        return []

    # Apply gamma correction to brighten dark areas (gamma > 1 brightens)
    gamma = 1.4
    inv_gamma = 1.0 / gamma  # Inverse for lookup table calculation
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')  # Create LUT
    enhanced = cv2.LUT(roi, table)  # Apply gamma correction

    # Convert to LAB color space and apply CLAHE to L channel for better contrast
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)  # Convert BGR to LAB
    l, a, b = cv2.split(lab)  # Split into channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    l = clahe.apply(l)  # Apply contrast enhancement to L channel
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)  # Merge back to BGR

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)  # Bilateral filter to reduce noise while preserving edges
    _, thresh_high = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)  # Threshold to find bright regions

    # Morphological operations to clean up noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Small kernel for opening
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Larger kernel for closing
    cleaned = cv2.morphologyEx(thresh_high, cv2.MORPH_OPEN, kernel_open)  # Remove small noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)  # Fill small gaps

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find bright blob contours
    roi_h, roi_w = roi.shape[:2]  # Get ROI dimensions
    max_area = roi_w * roi_h * 0.15  # Maximum area is 15% of ROI (filter out huge blobs)

    results = []  # List to store detected headlights
    for cnt in contours:  # Process each contour
        area = cv2.contourArea(cnt)  # Calculate contour area
        if not (min_area < area < max_area):  # Filter by area size
            continue
        hx, hy, hw, hh = cv2.boundingRect(cnt)  # Get bounding rectangle
        peri = cv2.arcLength(cnt, True)  # Calculate perimeter
        if peri <= 0:  # Skip if invalid perimeter
            continue
        circularity = 4.0 * np.pi * (area / (peri * peri))  # Calculate circularity (1.0 = perfect circle)
        aspect_ratio = hw / float(hh) if hh > 0 else 0  # Calculate width/height ratio
        if not (0.3 < circularity < 1.5 and 0.3 < aspect_ratio < 3.0):  # Filter by shape characteristics
            continue

        # Verify brightness in the region
        mask = np.zeros(gray.shape, dtype=np.uint8)  # Create mask
        cv2.fillPoly(mask, [cnt], 255)  # Fill contour area in mask
        mean_intensity = cv2.mean(gray, mask=mask)[0]  # Calculate mean brightness
        if mean_intensity <= 180:  # Filter out dim regions
            continue

        relative_y = hy / float(roi_h) if roi_h > 0 else 1.0  # Relative Y position in ROI
        if relative_y >= 0.8:  # Filter out headlights in bottom 20% (likely reflections)
            continue

        abs_x, abs_y = hx + x1, hy + y1  # Convert back to absolute frame coordinates
        results.append((abs_x, abs_y, hw, hh))  # Add to results list

    # Limit to maximum 3 headlights, keeping largest by area
    if len(results) > 3:
        results.sort(key=lambda r: r[2] * r[3], reverse=True)  # Sort by area (width * height)
        results = results[:3]  # Keep only top 3
    return results

def detect_headlights(frame, vehicle_bboxes):
    """Detect headlights within vehicle bounding boxes, with fallback to full-frame ROI"""
    headlights = []  # List to store all detected headlights
    for x1, y1, x2, y2 in vehicle_bboxes:  # Process each vehicle bounding box
        vw, vh = x2 - x1, y2 - y1  # Calculate vehicle width and height
        headlights.extend(_detect_headlights_in_bbox(frame, (x1, y1, vw, vh)))  # Detect headlights in vehicle ROI

    if not headlights:  # If no headlights found in vehicles, use fallback
        h, w = frame.shape[:2]  # Get frame dimensions
        y0 = int(h * HEADLIGHT_FALLBACK_ROI_TOP_FRAC)  # Calculate top Y coordinate (30% from top)
        headlights = _detect_headlights_in_bbox(frame, (0, y0, w, h - y0))  # Search bottom 70% of frame
    return headlights

def map_headlights_to_columns(headlights, fw):
    """Map detected headlight positions to column indices (0-15)"""
    blocked = [False] * TOTAL_COLUMNS  # Initialize all columns as unblocked
    for (hx, hy, hw, hh) in headlights:  # Process each detected headlight
        cx = hx + hw // 2  # Calculate headlight center X coordinate
        col = int((cx / fw) * TOTAL_COLUMNS)  # Convert X position to column index (0-15)
        col = max(0, min(TOTAL_COLUMNS - 1, col))  # Clamp to valid column range
        blocked[col] = True  # Block the column containing this headlight
        # Block adjacent columns if headlight is large (wider than 7% of frame)
        if hw > fw * 0.07 and col + 1 < TOTAL_COLUMNS:  # Block right adjacent column
            blocked[col + 1] = True
        if hw > fw * 0.07 and col - 1 >= 0:  # Block left adjacent column
            blocked[col - 1] = True
    return blocked

def draw_viz_16col(frame, vehicles, blocked_cols, column_cooldowns, fn, tf, vidx, total, vname):
    """Draw visualization overlay showing columns, vehicles, and status information"""
    h, w = frame.shape[:2]  # Get frame height and width
    col_width = w // TOTAL_COLUMNS  # Calculate width of each column
    
    # Draw 16 vertical grid lines to show column boundaries
    for i in range(1, TOTAL_COLUMNS):  # Draw lines between columns (skip first)
        x = i * col_width  # Calculate X position of line
        color = (80, 80, 80) if not blocked_cols[i] and not blocked_cols[i-1] else (0, 0, 255)  # Gray if unblocked, red if blocked
        cv2.line(frame, (x, 0), (x, h), color, 1)  # Draw vertical line
    
    # Draw bounding boxes around detected vehicles
    for v in vehicles:
        x1, y1, x2, y2 = v['bbox']  # Extract bounding box coordinates
        color = (0, 255, 0) if v['moving'] else (100, 100, 100)  # Green if moving, gray if stationary
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Draw rectangle
        # Suppress text over green boxes as requested
    
    # Highlight blocked columns with red overlay
    for col in range(TOTAL_COLUMNS):
        if blocked_cols[col] or column_cooldowns[col] > 0:  # If column is blocked or in cooldown
            x1 = col * col_width  # Left edge of column
            x2 = (col + 1) * col_width  # Right edge of column
            
            overlay = frame.copy()  # Create overlay copy
            cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 0, 255), -1)  # Fill column with red
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # Blend red overlay at 20% opacity
            
            # Draw column number at top of column
            cv2.putText(frame, str(col), (x1 + col_width//2 - 8, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw cooldown timer if column is in cooldown period
            if column_cooldowns[col] > 0:
                cv2.putText(frame, f"{column_cooldowns[col]:.1f}s", 
                           (x1 + 5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    # Create info panel with system status
    info_h, info_w = 160, 350  # Info panel dimensions
    info_box = np.zeros((info_h, info_w, 3), dtype=np.uint8)  # Create black background
    
    cv2.putText(info_box, "16-COLUMN ADAPTIVE", (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Title
    cv2.putText(info_box, f"Video {vidx}/{total}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)  # Video counter
    cv2.putText(info_box, vname[:28], (10, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)  # Video filename
    cv2.putText(info_box, f"Vehicles: {len(vehicles)}", (10, 125), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Vehicle count
    cv2.putText(info_box, f"Blocked: {sum(blocked_cols)}/16", (10, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)  # Blocked column count
    
    # Blend info panel onto frame
    frame[10:10+info_h, 10:10+info_w] = cv2.addWeighted(
        frame[10:10+info_h, 10:10+info_w], 0.3, info_box, 0.7, 0)
    
    # Draw progress bar at bottom of frame
    prog = fn / tf if tf > 0 else 0  # Calculate progress (0.0 to 1.0)
    bar_w = w - 40  # Progress bar width
    bar_f = int(bar_w * prog)  # Filled portion width
    cv2.rectangle(frame, (20, h-40), (w-20, h-15), (50, 50, 50), -1)  # Draw background bar
    cv2.rectangle(frame, (20, h-40), (20+bar_f, h-15), (0, 255, 0), -1)  # Draw filled portion
    cv2.putText(frame, f"{fn}/{tf}", (w-150, h-45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Frame counter text
    
    return frame

def process_video(vpath, vidx, total):
    """Process a single video file with vehicle and headlight detection"""
    global vehicle_tracker
    vehicle_tracker = {}  # Reset tracker for each video
    
    vname = os.path.basename(vpath)  # Extract filename from path
    print(f"\nProcessing {vidx}/{total}: {vname}")
    
    cap = cv2.VideoCapture(vpath)  # Open video file
    if not cap.isOpened():  # Check if video opened successfully
        return 'skip'
    
    tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Frames per second (default 30)
    frame_delay = max(1, int(1000 / fps))  # Calculate delay in milliseconds for FPS sync
    
    fn = 0  # Current frame number
    paused = False  # Pause state
    
    column_last_detected = [0] * TOTAL_COLUMNS  # Timestamp of last detection for each column
    
    while True:
        if not paused:  # Only process frames when not paused
            ret, frame = cap.read()  # Read next frame from video
            if not ret:  # Break if video ended or read failed
                break
            
            fn += 1  # Increment frame counter
            current_time = time.time()  # Get current timestamp
            
            # Run YOLO vehicle detection on current frame
            results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=VEHICLE_CLASSES, verbose=False)
            
            detections = []  # List to store vehicle detections
            for result in results:  # Process YOLO results
                for box in result.boxes:  # Process each detected box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Extract coordinates from GPU
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),  # Convert to integers
                        'class_name': model.names[int(box.cls[0].cpu().numpy())]  # Get class name
                    })
            
            # Analyze vehicles and map to 16 columns
            vehicles, blocked_cols = analyze_vehicles_16col(detections, fw, fh)

            # Detect headlights using vehicle bounding boxes, with fallback to full-frame ROI
            vehicle_bboxes = [d['bbox'] for d in detections]  # Extract vehicle bounding boxes
            headlights = detect_headlights(frame, vehicle_bboxes)  # Detect headlights
            hl_blocked = map_headlights_to_columns(headlights, fw)  # Map headlights to columns

            # Merge vehicle and headlight blocking: block if either detects something
            for col in range(TOTAL_COLUMNS):
                blocked_cols[col] = blocked_cols[col] or hl_blocked[col]  # OR operation
            
            # Update timestamp for columns that are currently blocked
            for col in range(TOTAL_COLUMNS):
                if blocked_cols[col]:  # If column is blocked
                    column_last_detected[col] = current_time  # Update last detection time
            
            # Calculate remaining cooldown time for each column
            column_cooldowns = [max(0, COOLDOWN_SECONDS - (current_time - column_last_detected[col])) 
                               for col in range(TOTAL_COLUMNS)]
            
            # Determine final blocking state: active detection OR in cooldown period
            final_blocked = [blocked_cols[col] or column_cooldowns[col] > 0 
                            for col in range(TOTAL_COLUMNS)]
            
            # Send blocking command to Arduino (format: "BLOCK:0,0,1,1,0,...")
            cmd = "BLOCK:" + ",".join([str(1 if b else 0) for b in final_blocked])  # Convert to string
            send_cmd(cmd)  # Send via serial
            
            # Draw visualization overlay on frame
            display = draw_viz_16col(frame.copy(), vehicles, blocked_cols, 
                                    column_cooldowns, fn, tf, vidx, total, vname)
            
            # Show pause message overlay if video is paused
            if paused:
                cv2.putText(display, "PAUSED - Press SPACE to resume", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('16-Column Adaptive Headlight', display)  # Display frame
        
        key = cv2.waitKey(frame_delay) & 0xFF  # Wait for keypress and get key code
        if key == 27 or key == ord('q'):  # ESC or 'q' key: quit
            cap.release()  # Release video capture
            return 'quit'
        elif key == ord('n'):  # 'n' key: skip to next video
            cap.release()
            return 'skip'
        elif key == ord(' '):  # SPACE key: toggle pause
            paused = not paused
        elif key == ord('r'):  # 'r' key: restart video from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to frame 0
            fn = 0  # Reset frame counter
    
    cap.release()  # Release video capture
    return 'complete'  # Video processing completed

# ============ MAIN ============
print("\nStarting 16-column precision detection...\n")

# Process each video in the list
for i, v in enumerate(videos, 1):  # Enumerate starting from 1
    if process_video(v, i, len(videos)) == 'quit':  # Process video, break if user quits
        break

send_cmd("CLEAR")  # Send clear command to Arduino to reset all columns
cv2.destroyAllWindows()  # Close all OpenCV windows
if arduino:  # Close serial connection if Arduino was connected
    arduino.close()

print("\n✓ Complete!")  # Print completion message
