#include <FastLED.h>

// ============ CONFIGURATION ============
#define NUM_LEDS_PER_MATRIX 64
#define LEFT_DATA_PIN 6
#define RIGHT_DATA_PIN 9
#define MATRIX_WIDTH 8
#define MATRIX_HEIGHT 8
#define TOTAL_COLUMNS 16  // 8 per matrix

// Lower brightness levels
#define HIGH_BEAM_ON 15   // High beam brightness (was 255)
#define HIGH_BEAM_OFF 0   // Column blocked (OFF)
#define LOW_BEAM 7        // Low beam brightness (was 10)

// Timing
#define COOLDOWN_TIME 3000

// LED Arrays
CRGB leftLEDs[NUM_LEDS_PER_MATRIX];
CRGB rightLEDs[NUM_LEDS_PER_MATRIX];

// Column blocking state (16 columns total: 0-7 left, 8-15 right)
bool blockedColumns[TOTAL_COLUMNS] = {false};
unsigned long columnBlockTime[TOTAL_COLUMNS] = {0};

// ============ SETUP ============
void setup() {
  Serial.begin(9600);
  
  // Initialize matrices
  FastLED.addLeds<WS2812B, LEFT_DATA_PIN, GRB>(leftLEDs, NUM_LEDS_PER_MATRIX);
  FastLED.addLeds<WS2812B, RIGHT_DATA_PIN, GRB>(rightLEDs, NUM_LEDS_PER_MATRIX);
  
  // Start with full beam
  setFullBeam();
  FastLED.show();
  
  Serial.println("READY");
  Serial.println("========================================");
  Serial.println("ADAPTIVE HEADLIGHT - 16 COLUMN PRECISION");
  Serial.println("========================================");
  Serial.println("16 columns total (8 left + 8 right)");
  Serial.println("High beam: 15, Low beam: 7");
  Serial.println("Blocks specific columns only");
  Serial.println("");
  Serial.println("Commands:");
  Serial.println("  BLOCK:0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0");
  Serial.println("  CLEAR");
  Serial.println("  STATUS");
  Serial.println("  TEST");
  Serial.println("========================================");
}

// ============ MAIN LOOP ============
void loop() {
  // Handle commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    handleCommand(cmd);
  }
  
  // Auto-restore blocked columns after cooldown
  unsigned long currentTime = millis();
  bool updated = false;
  
  for (int col = 0; col < TOTAL_COLUMNS; col++) {
    if (blockedColumns[col] && (currentTime - columnBlockTime[col]) > COOLDOWN_TIME) {
      blockedColumns[col] = false;
      updated = true;
    }
  }
  
  if (updated) {
    updateHeadlights();
    Serial.println("-> Columns restored after cooldown");
  }
  
  delay(100);
}

// ============ COMMAND HANDLER ============
void handleCommand(String cmd) {
  if (cmd.startsWith("BLOCK:")) {
    parseBlockCommand(cmd.substring(6));
    
  } else if (cmd == "CLEAR") {
    for (int i = 0; i < TOTAL_COLUMNS; i++) {
      blockedColumns[i] = false;
    }
    setFullBeam();
    Serial.println("-> All columns cleared");
    
  } else if (cmd == "TEST") {
    runTest();
    
  } else if (cmd == "STATUS") {
    printStatus();
  }
}

// ============ PARSE BLOCK COMMAND ============
void parseBlockCommand(String data) {
  // Format: BLOCK:0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  // 16 values (0 or 1) for each column
  
  unsigned long currentTime = millis();
  
  int startPos = 0;
  for (int col = 0; col < TOTAL_COLUMNS; col++) {
    int commaPos = data.indexOf(',', startPos);
    String value;
    
    if (commaPos > 0 && col < TOTAL_COLUMNS - 1) {
      value = data.substring(startPos, commaPos);
      startPos = commaPos + 1;
    } else {
      value = data.substring(startPos);
    }
    
    if (value == "1") {
      blockedColumns[col] = true;
      columnBlockTime[col] = currentTime;
    } else {
      blockedColumns[col] = false;
    }
  }
  
  updateHeadlights();
  
  // Print which columns are blocked
  Serial.print("Blocked columns: ");
  bool anyBlocked = false;
  for (int i = 0; i < TOTAL_COLUMNS; i++) {
    if (blockedColumns[i]) {
      Serial.print(i);
      Serial.print(" ");
      anyBlocked = true;
    }
  }
  if (!anyBlocked) {
    Serial.print("None");
  }
  Serial.println();
}

// ============ SET FULL BEAM ============
void setFullBeam() {
  // LEFT MATRIX (Columns 0-7)
  for (int row = 0; row < MATRIX_HEIGHT; row++) {
    for (int col = 0; col < MATRIX_WIDTH; col++) {
      int led = row * MATRIX_WIDTH + col;
      
      if (row < 3) {
        // Top 3 rows: High beam
        leftLEDs[led] = CRGB(HIGH_BEAM_ON, HIGH_BEAM_ON, HIGH_BEAM_ON);
      } else {
        // Bottom 5 rows: Low beam
        leftLEDs[led] = CRGB(LOW_BEAM, LOW_BEAM, LOW_BEAM);
      }
    }
  }
  
  // RIGHT MATRIX (Columns 8-15)
  for (int row = 0; row < MATRIX_HEIGHT; row++) {
    for (int col = 0; col < MATRIX_WIDTH; col++) {
      int led = row * MATRIX_WIDTH + col;
      
      if (row < 3) {
        rightLEDs[led] = CRGB(HIGH_BEAM_ON, HIGH_BEAM_ON, HIGH_BEAM_ON);
      } else {
        rightLEDs[led] = CRGB(LOW_BEAM, LOW_BEAM, LOW_BEAM);
      }
    }
  }
  
  FastLED.show();
}

// ============ UPDATE HEADLIGHTS ============
void updateHeadlights() {
  // LEFT MATRIX (Columns 0-7)
  for (int row = 0; row < MATRIX_HEIGHT; row++) {
    for (int col = 0; col < MATRIX_WIDTH; col++) {
      int led = row * MATRIX_WIDTH + col;
      int globalCol = col;  // Columns 0-7 for left matrix
      
      if (row < 3) {
        // Top 3 rows: Check if column is blocked
        uint8_t brightness = blockedColumns[globalCol] ? HIGH_BEAM_OFF : HIGH_BEAM_ON;
        leftLEDs[led] = CRGB(brightness, brightness, brightness);
      } else {
        // Bottom 5 rows: Always low beam
        leftLEDs[led] = CRGB(LOW_BEAM, LOW_BEAM, LOW_BEAM);
      }
    }
  }
  
  // RIGHT MATRIX (Columns 8-15)
  for (int row = 0; row < MATRIX_HEIGHT; row++) {
    for (int col = 0; col < MATRIX_WIDTH; col++) {
      int led = row * MATRIX_WIDTH + col;
      int globalCol = 8 + col;  // Columns 8-15 for right matrix
      
      if (row < 3) {
        uint8_t brightness = blockedColumns[globalCol] ? HIGH_BEAM_OFF : HIGH_BEAM_ON;
        rightLEDs[led] = CRGB(brightness, brightness, brightness);
      } else {
        rightLEDs[led] = CRGB(LOW_BEAM, LOW_BEAM, LOW_BEAM);
      }
    }
  }
  
  FastLED.show();
}

// ============ TEST FUNCTION ============
void runTest() {
  Serial.println("========================================");
  Serial.println("TESTING 16-COLUMN BLOCKING");
  Serial.println("========================================");
  
  // Test 1: Block column 0 (leftmost of left matrix)
  Serial.println("1. Block column 0 (left edge)");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  blockedColumns[0] = true;
  updateHeadlights();
  delay(1500);
  
  // Test 2: Block column 7 (rightmost of left matrix)
  Serial.println("2. Block column 7 (left-center)");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  blockedColumns[7] = true;
  updateHeadlights();
  delay(1500);
  
  // Test 3: Block column 8 (leftmost of right matrix)
  Serial.println("3. Block column 8 (right-center)");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  blockedColumns[8] = true;
  updateHeadlights();
  delay(1500);
  
  // Test 4: Block column 15 (rightmost of right matrix)
  Serial.println("4. Block column 15 (right edge)");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  blockedColumns[15] = true;
  updateHeadlights();
  delay(1500);
  
  // Test 5: Block multiple columns
  Serial.println("5. Block columns 3,4,5 (left matrix)");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  blockedColumns[3] = true;
  blockedColumns[4] = true;
  blockedColumns[5] = true;
  updateHeadlights();
  delay(2000);
  
  // Test 6: Block right matrix columns
  Serial.println("6. Block columns 10,11,12 (right matrix)");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  blockedColumns[10] = true;
  blockedColumns[11] = true;
  blockedColumns[12] = true;
  updateHeadlights();
  delay(2000);
  
  // Clear all
  Serial.println("7. Clear all - Full beam");
  for (int i = 0; i < TOTAL_COLUMNS; i++) blockedColumns[i] = false;
  setFullBeam();
  
  Serial.println("Test complete!");
  Serial.println("========================================");
}

// ============ PRINT STATUS ============
void printStatus() {
  Serial.println("========================================");
  Serial.println("16-COLUMN STATUS:");
  
  // Left matrix
  Serial.print("Left (0-7):  ");
  for (int i = 0; i < 8; i++) {
    Serial.print(blockedColumns[i] ? "X" : "O");
  }
  Serial.println();
  
  // Right matrix
  Serial.print("Right (8-15): ");
  for (int i = 8; i < 16; i++) {
    Serial.print(blockedColumns[i] ? "X" : "O");
  }
  Serial.println();
  
  Serial.println("O = ON, X = OFF");
  Serial.println("========================================");
}
