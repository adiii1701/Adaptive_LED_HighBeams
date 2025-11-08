# Adaptive Headlight Control System

An intelligent adaptive headlight control system that uses computer vision to detect oncoming vehicles and automatically dims specific columns of high beams to prevent glare while maintaining maximum visibility.

## 🚗 Features

- **16-Column Precision Control**: Independently controls 16 columns (8 per LED matrix) for granular headlight adjustment
- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection (cars, motorcycles, buses, trucks)
- **Headlight Detection**: Advanced image processing to detect vehicle headlights in low-light conditions
- **Motion Tracking**: Tracks vehicles across frames to determine movement and persistence
- **Cooldown System**: Maintains column blocking for 2 seconds after last detection to prevent flickering
- **Arduino Integration**: Serial communication with Arduino to control WS2812B LED matrices
- **Video Processing**: Process video files or real-time webcam feed
- **Visual Feedback**: Real-time visualization with vehicle bounding boxes, blocked columns, and system status

## 🛠️ Hardware Requirements

- **Arduino Board** (Uno, Nano, or compatible)
- **2x WS2812B LED Matrices** (8x8, 64 LEDs each)
- **USB Cable** for Arduino connection
- **Camera** (webcam or video files for testing)

## 📋 Software Requirements

- Python 3.7+
- Arduino IDE
- Required Python packages (see Installation)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd demo
```

### 2. Install Python Dependencies

```bash
cd Python
pip install -r requirements.txt
```

### 3. Download YOLO Model

The YOLOv8 model (`yolov8n.pt`) should be placed in the `Python/models/` directory. If not present, it will be automatically downloaded on first run.

### 4. Upload Arduino Sketch

1. Open `Arduino/sketch_nov3a/sketch_nov3a.ino` in Arduino IDE
2. Install the FastLED library (Tools → Manage Libraries → Search "FastLED")
3. Select your Arduino board and port
4. Upload the sketch to your Arduino
5. headlight_control.py is the final working project demo file w.r.t sketch_nov3a.ino

## 🚀 Usage

### Main Headlight Control System

```bash
cd Python
python headlight_control.py [--video path/to/video]
```

**Options:**
- `--video`: Process a specific video file (default: processes all videos in `videos/` folder)

**Controls:**
- `SPACE`: Pause/Resume video
- `R`: Restart video from beginning
- `N`: Skip to next video
- `ESC` or `Q`: Quit


## ⚙️ Configuration

### Python Configuration (`headlight_control.py`)

Key parameters can be adjusted in the configuration section:

```python
ARDUINO_PORT = '/dev/tty.usbserial-A5069RR4'  # Serial port (auto-detect on Linux/Mac)
```

### Arduino Configuration (`sketch_nov3a.ino`)

```cpp
#define NUM_LEDS_PER_MATRIX 64    // LEDs per matrix
#define LEFT_DATA_PIN 6            // Left matrix data pin
#define RIGHT_DATA_PIN 9           // Right matrix data pin
#define HIGH_BEAM_ON 15            // High beam brightness
#define LOW_BEAM 7                 // Low beam brightness
```

## 🔌 Serial Communication Protocol

The system uses a simple text-based protocol:

**Block Command:**
```
BLOCK:0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0
```
- 16 comma-separated values (0 or 1)
- Each value corresponds to a column (0-15)
- `1` = block column (dim), `0` = unblock (full beam)

**Clear Command:**
```
CLEAR
```
- Resets all columns to unblocked state

**Status Command:**
```
STATUS
```
- Returns current blocking state

## 🎯 How It Works

1. **Video Capture**: System captures frames from video file or webcam
2. **Vehicle Detection**: YOLOv8 detects vehicles (cars, motorcycles, buses, trucks) in the frame
3. **Motion Analysis**: Tracks vehicles across frames to determine if they're moving
4. **Headlight Detection**: Uses image processing (gamma correction, CLAHE, thresholding) to detect bright headlights
5. **Column Mapping**: Maps detected vehicles/headlights to 16 columns based on horizontal position
6. **Arduino Control**: Sends blocking commands via serial to Arduino
7. **LED Control**: Arduino dims specific columns on WS2812B matrices



---

**Note**: This project is for educational and demonstration purposes. Always follow local traffic laws and regulations when implementing adaptive headlight systems in vehicles.

