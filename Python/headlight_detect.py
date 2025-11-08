import cv2
import numpy as np
import argparse
import time
from collections import deque

# Parse command line arguments for headlight detection parameters
parser = argparse.ArgumentParser(description='Headlight detection from webcam or video file')
parser.add_argument('--video', default=None, help='Path to video file (defaults to webcam)')
parser.add_argument('--threshold', type=int, default=220, help='Brightness threshold (0-255)')
parser.add_argument('--adaptive', action='store_true', help='Use adaptive thresholding (better for varying lighting)')
parser.add_argument('--adaptive-block', type=int, default=31, help='Adaptive threshold block size (odd)')
parser.add_argument('--adaptive-C', type=int, default=-5, help='Adaptive threshold constant C')
parser.add_argument('--min-area', type=int, default=80, help='Minimum contour area to keep (pixels after ROI)')
parser.add_argument('--max-area-frac', type=float, default=0.15, help='Max area fraction relative to ROI to avoid huge blobs')
parser.add_argument('--gamma', type=float, default=1.3, help='Gamma correction (>1 brightens)')
parser.add_argument('--use-clahe', action='store_true', help='Apply CLAHE to V channel for contrast')
parser.add_argument('--roi', type=float, default=0.3, help='Fraction from top to ignore (0=full frame, 0.3 uses bottom 70%)')
parser.add_argument('--persist', type=int, default=3, help='Frames required for persistence (temporal smoothing)')
parser.add_argument('--open-k', type=int, default=3, help='Morph open kernel size')
parser.add_argument('--close-k', type=int, default=5, help='Morph close kernel size')
parser.add_argument('--blob', action='store_true', help='Use LAB L-channel + SimpleBlobDetector (night headlights)')
parser.add_argument('--blob-min-area-frac', type=float, default=0.0003, help='Min blob area fraction relative to ROI')
parser.add_argument('--blob-max-area-frac', type=float, default=0.08, help='Max blob area fraction relative to ROI')
parser.add_argument('--blob-min-circularity', type=float, default=0.6, help='Min circularity for blobs')
parser.add_argument('--blob-min-inertia', type=float, default=0.4, help='Min inertia ratio for blobs')
parser.add_argument('--blob-min-convexity', type=float, default=0.6, help='Min convexity for blobs')
parser.add_argument('--blob-threshold-step', type=float, default=5.0, help='Internal threshold step for blob detector')
parser.add_argument('--blob-min-threshold', type=float, default=220.0, help='Internal min threshold for blob detector')
parser.add_argument('--blob-max-threshold', type=float, default=255.0, help='Internal max threshold for blob detector')
parser.add_argument('--fps-sync', action='store_true', help='Sync playback to video FPS')
args = parser.parse_args()

# Video source configuration
SOURCE_MODE = 1  # 0 = webcam, 1 = video file
VIDEO_PATH = "/Users/adityagadagandla/Desktop/demo/test.mp4"

# Initialize video capture
if args.video is not None:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0 if SOURCE_MODE == 0 else VIDEO_PATH)

print("Starting headlight detection demo... Press ESC to exit.")

# Setup FPS synchronization
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_delay = max(1, int(1000 / fps)) if args.fps_sync else 1
print(f"Video FPS: {fps:.1f}, Frame delay: {frame_delay}ms")

# Initialize temporal smoothing buffer
history = deque(maxlen=max(1, args.persist))

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Crop region of interest to focus on road area (ignore sky/signs)
    roi_top_frac = max(0.0, min(0.9, args.roi))
    h_full, w_full = frame.shape[:2]
    y0 = int(h_full * roi_top_frac)
    roi = frame[y0:h_full, :]

    # Image enhancement for better headlight detection in low light
    gamma = max(0.1, min(3.0, args.gamma))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
    roi_gamma = cv2.LUT(roi, table)

    # Optional CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if args.use_clahe:
        hsv = cv2.cvtColor(roi_gamma, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v_ch)
        hsv_eq = cv2.merge([h_ch, s_ch, v_eq])
        roi_pre = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    else:
        roi_pre = roi_gamma

    # Convert to grayscale for processing
    gray = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Branch A: LAB L-channel + SimpleBlobDetector (requested method)
    if args.blob:
        lab = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        l_blur = cv2.GaussianBlur(l_channel, (25, 25), 0)

        h_roi, w_roi = l_blur.shape[:2]
        image_area = float(h_roi * w_roi)
        min_area = args.blob_min_area_frac * image_area
        max_area = args.blob_max_area_frac * image_area

        params = cv2.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = min(h_roi, w_roi) * 0.02
        params.thresholdStep = args.blob_threshold_step
        params.minThreshold = args.blob_min_threshold
        params.maxThreshold = args.blob_max_threshold
        params.minRepeatability = 2

        params.filterByArea = True
        params.minArea = max(1.0, min_area)
        params.maxArea = max_area

        params.filterByColor = True
        params.blobColor = 255

        params.filterByCircularity = True
        params.minCircularity = args.blob_min_circularity
        params.maxCircularity = 0.95

        params.filterByInertia = True
        params.minInertiaRatio = args.blob_min_inertia
        params.maxInertiaRatio = 0.9

        params.filterByConvexity = True
        params.minConvexity = args.blob_min_convexity
        params.maxConvexity = 1.0

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(l_blur)

        # Additional filtering for reflections: check brightness and position
        filtered_keypoints = []
        for kp in keypoints:
            x, y = kp.pt
            cx, cy = int(x), int(y)
            
            # Check brightness in original gray image
            if cx < gray.shape[1] and cy < gray.shape[0]:
                brightness = gray[cy, cx]
                # Only accept very bright regions (headlights are typically >200)
                if brightness < 200:
                    continue
            
            # Position filter: headlights are typically in upper-middle area of ROI
            roi_h, roi_w = l_blur.shape[:2]
            rel_y = cy / roi_h
            if rel_y > 0.7:  # Avoid bottom 30% (likely reflections from road/vehicles)
                continue
            
            # Size filter: avoid tiny specks and huge flares
            r = int(max(2, kp.size / 2))
            if r < 3 or r > 25:  # reasonable headlight size range
                continue
                
            filtered_keypoints.append(kp)

        # Draw filtered keypoints mapped back to full frame coordinates
        for kp in filtered_keypoints:
            x, y = kp.pt
            r = int(max(2, kp.size / 2))
            cx = int(x)
            cy = int(y) + y0
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 2)
            # cv2.putText(frame, 'Headlight', (cx, max(0, cy - r - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show intermediate visualization (optional):
        # im_with_keypoints = cv2.drawKeypoints(roi_pre, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('Blob ROI', im_with_keypoints)

    # Branch B: Threshold + morphology + temporal smoothing (default)
    else:
        # Threshold for bright regions
        if args.adaptive:
            blk = args.adaptive_block if args.adaptive_block % 2 == 1 else args.adaptive_block + 1
            blk = max(3, blk)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, blk, args.adaptive_C)
        else:
            _, thresh = cv2.threshold(blurred, args.threshold, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        open_k = max(1, args.open_k)
        close_k = max(1, args.close_k)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # Temporal persistence: require presence in N recent frames
        history.append(cleaned)
        if len(history) == 1:
            stable = cleaned
        else:
            acc = np.zeros_like(cleaned, dtype=np.uint16)
            for m in history:
                acc = acc + (m > 0).astype(np.uint16)
            stable = (acc >= max(1, args.persist - 1)).astype(np.uint8) * 255

        # Find contours on stabilized mask
        contours, _ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Filter by area (avoid small noise), and avoid huge blobs
        area = cv2.contourArea(cnt)
        h_roi, w_roi = roi_pre.shape[:2]
        if area < args.min_area or area > args.max_area_frac * (w_roi * h_roi):
            continue

        # Shape heuristics: circularity and aspect ratio
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * (area / (peri * peri))
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if circularity < 0.2 or circularity > 1.5:
            continue
        if aspect < 0.2 or aspect > 5.0:
            continue

        # Map ROI box back to full frame
        y_full = y + y0
        cv2.rectangle(frame, (x, y_full), (x + w, y_full + h), (0, 0, 255), 2)
        # cv2.putText(frame, "Headlight", (x, max(0, y_full - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Headlight Detection Demo", frame)
    
    # Exit on ESC with FPS sync
    if cv2.waitKey(frame_delay) == 27:
        break

    # Optional: sleep for more accurate timing (uncomment if needed)
    # if args.fps_sync:
    #     time.sleep(max(0, (1.0 / fps) - 0.001))

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Headlight detection demo ended.")