import serial
import time
import argparse
import glob

# Function to automatically find Arduino serial port (optional)
def find_serial_port(explicit_port: str | None = None) -> str | None:
    if explicit_port:
        return explicit_port
    candidates = []
    # Search for common Arduino port patterns
    candidates.extend(glob.glob('/dev/tty.usbmodem*'))
    candidates.extend(glob.glob('/dev/tty.usbserial*'))
    candidates.extend(glob.glob('/dev/ttyACM*'))
    candidates.extend(glob.glob('/dev/ttyUSB*'))
    if not candidates:
        return None
    return candidates[0]

# Parse command line arguments for serial testing
parser = argparse.ArgumentParser(description='Send serial test signals to Arduino')
parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
parser.add_argument('--baud', type=int, default=9600)
parser.add_argument('--cycles', type=int, default=5)
parser.add_argument('--on-sec', type=float, default=1.0)
parser.add_argument('--off-sec', type=float, default=1.0)
args = parser.parse_args()

# Setup serial connection parameters
PORT = find_serial_port(args.port)
BAUD_RATE = args.baud

try:
    if PORT is None:
        raise serial.SerialException("No serial port detected")
    # Connect to Arduino via serial
    arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    print(f"Connected to Arduino on {PORT}.")

    # Send test signals to verify Arduino communication
    print(f"Sending test signals ({args.cycles} cycles: '1' then '0')...")
    
    for i in range(args.cycles):
        arduino.write(b'1')  # Send '1' to simulate vehicle detection (dim high beams)
        print(f"Cycle {i+1}: Sent '1' (LED ON)")
        time.sleep(args.on_sec)
        arduino.write(b'0')  # Send '0' to simulate no detection (full high beams)
        print(f"Cycle {i+1}: Sent '0' (LED OFF)")
        time.sleep(args.off_sec)

except serial.SerialException as e:
    print(f"Error: Could not connect to {PORT}. {e}")
finally:
    # Clean up serial connection
    if 'arduino' in locals():
        arduino.close()
        print("Serial connection closed.")