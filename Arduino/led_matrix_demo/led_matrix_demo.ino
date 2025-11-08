// Arduino sketch for LED matrix control via serial communication
// Requires LedControl library (install from Arduino IDE Library Manager)
// '1' = Dim pattern (simulates adaptive headlight dimming)
// '0' = Full bright pattern (simulates full high beams)

#include <LedControl.h>

// MAX7219 LED matrix pin configuration
LedControl lc = LedControl(12, 10, 11, 1);  // DIN=12, CS=11, CLK=10, 1 matrix

void setup() {
  Serial.begin(9600);
  
  // Initialize LED matrix
  lc.shutdown(0, false);       // Wake up the matrix
  lc.setIntensity(0, 8);       // Set brightness (0-15)
  lc.clearDisplay(0);          // Clear all LEDs
  
  // Display test pattern on startup
  fullBright();
}

void loop() {
  // Process incoming serial commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      dimPattern();  // Show dimmed pattern (adaptive headlights)
    } else if (cmd == '0') {
      fullBright();  // Show full bright pattern (high beams)
    }
  }
}

// Display full bright pattern (all LEDs on)
void fullBright() {
  for (int row = 0; row < 8; row++) {
    lc.setRow(0, row, B11111111);  // Turn on all LEDs in row
  }
}

// Display dimmed pattern (top half on, bottom half off)
void dimPattern() {
  // Turn on top half (rows 0-3)
  for (int row = 0; row < 4; row++) {
    lc.setRow(0, row, B11111111);  // All LEDs on
  }
  // Turn off bottom half (rows 4-7)
  for (int row = 4; row < 8; row++) {
    lc.setRow(0, row, B00000000);  // All LEDs off
  }
}