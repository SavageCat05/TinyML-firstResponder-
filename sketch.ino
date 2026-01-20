#define LED_PIN 2
#define BUZZER_PIN 15

#define BUZZER_FREQ 2000      // 2 kHz
#define BUZZER_RESOLUTION 8   // 8-bit PWM

String inputCommand = "";

void beep(int onTime, int offTime, int repeatCount) {
  for (int i = 0; i < repeatCount; i++) {
    ledcWriteTone(BUZZER_PIN, BUZZER_FREQ);  // Sound ON
    delay(onTime);
    ledcWriteTone(BUZZER_PIN, 0);            // Sound OFF
    delay(offTime);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);

  // ✅ CORRECT for ESP32 core 3.x
  ledcAttach(BUZZER_PIN, BUZZER_FREQ, BUZZER_RESOLUTION);

  digitalWrite(LED_PIN, LOW);
  ledcWriteTone(BUZZER_PIN, 0);

  Serial.println("ESP32 SOS Simulation Started");
  Serial.println("Type: HELP, EMERGENCY, NORMAL, STOP");
}

void loop() {
  if (Serial.available()) {
    inputCommand = Serial.readStringUntil('\n');
    inputCommand.trim();
    inputCommand.toUpperCase();

    if (inputCommand == "HELP") {
      digitalWrite(LED_PIN, HIGH);

      Serial.println("🚨 HELP detected!");
      Serial.println("📞 Calling Police...");

      // Fast alarm
      beep(200, 100, 5);

    } else if (inputCommand == "EMERGENCY") {
      digitalWrite(LED_PIN, LOW);

      Serial.println("🚨 EMERGENCY detected!");
      Serial.println("📞 Calling Ambulance...");

      // Slow alarm
      beep(400, 300, 3);

    } else if (inputCommand == "NORMAL") {
      digitalWrite(LED_PIN, LOW);
      ledcWriteTone(BUZZER_PIN, 0);

      Serial.println("✅ Normal state. No action.");

    } else if (inputCommand == "STOP") {
      digitalWrite(LED_PIN, LOW);
      ledcWriteTone(BUZZER_PIN, 0);

      Serial.println("🛑 System stopped.");

    } else {
      Serial.println("❓ Unknown command.");
    }
  }
}
