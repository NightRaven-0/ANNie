// robot_ann.ino
// Arduino Uno sketch (L298N, HC-SR04 on servo), ANN inference using weights from models/arduino_weights.h
#include <Servo.h>
#include <Arduino.h>

// --- pins (same as earlier sketch) ---
const uint8_t ENA = 5;  //motor pins
const uint8_t ENB = 6;  //motor pins
const uint8_t IN1 = 7;  //motor pins
const uint8_t IN2 = 8;  //motor pins
const uint8_t IN3 = 9;  //motor pins
const uint8_t IN4 = 10;  //motor pins
const uint8_t TRIG = 12;  //ultrasonic sensor
const uint8_t ECHO = 11;  //ultrasonic sensor
const uint8_t FRONT_LED = 2;  //leds
const uint8_t BACK_LED = 4;   //leds
const uint8_t SERVO_PIN = 3;  //servo

const int SAFE_DISTANCE = 25;  //cm  
const int CRITICAL_DISTANCE = 10;  //cm
const int SERVO_CENTER = 90;  
const int SERVO_LEFT = 150;
const int SERVO_RIGHT = 30;

Servo panServo;

void motorsStop() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0); analogWrite(ENB, 0);
  digitalWrite(FRONT_LED, LOW); digitalWrite(BACK_LED, LOW);
}
void motorsForward(uint8_t s) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, s); analogWrite(ENB, s);
  digitalWrite(FRONT_LED, HIGH); digitalWrite(BACK_LED, LOW);
}
void motorsBackward(uint8_t s) {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  analogWrite(ENA, s); analogWrite(ENB, s);
  digitalWrite(FRONT_LED, LOW); digitalWrite(BACK_LED, HIGH);
}
void motorsTurnLeft(uint8_t s) {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, s); analogWrite(ENB, s);
}
void motorsTurnRight(uint8_t s) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  analogWrite(ENA, s); analogWrite(ENB, s);
}

// --- include generated weights header ---
// The trainer will write ANNie/models/arduino_weights.h
// For now you can create a placeholder file there or run the trainer to generate it.
#include "../models/arduino_weights.h" // ensure correct relative path in Arduino IDE (or copy header to sketch folder)


// If the header defines the arrays W1,B1,W2,B2 and shapes, the code below will use them.
// For this example, we'll assume shapes: W1 = [H1 x 3], B1 = [H1], W2 = [4 x H1], B2=[4]
// If your generated header uses a different naming, adapt the names here.

#ifndef W1 // if the weights header is missing, define tiny example weights
// Example tiny model: 3 -> 8 -> 4
const int H1 = 8;
const float W1_example[8*3] = {
  0.12f,-0.10f,0.06f,
  -0.08f,0.15f,-0.05f,
  0.20f,-0.12f,-0.02f,
  -0.05f,0.10f,0.12f,
  0.07f,0.02f,-0.09f,
  -0.15f,0.18f,0.04f,
  0.09f,-0.06f,0.11f,
  0.03f,0.05f,-0.14f
};
const float B1_example[8] = {0.01f,-0.02f,0.00f,0.03f,-0.01f,0.02f,0.00f,-0.02f};
const float W2_example[4*8] = {
  0.18f,-0.12f,0.05f,0.02f,0.01f,-0.07f,0.10f,0.03f,
  -0.09f,0.14f,-0.04f,0.12f,-0.05f,0.08f,-0.02f,-0.06f,
  -0.07f,0.06f,-0.11f,-0.03f,0.15f,-0.02f,-0.08f,0.10f,
  0.02f,-0.01f,0.03f,-0.06f,0.04f,0.12f,-0.05f,-0.02f
};
const float B2_example[4] = {0.01f,-0.01f,0.00f,0.02f};
#define H1_DEFINED 1
#endif

// choose which arrays to use (generated header should define W1,B1,W2,B2)
// If generated header uses different names, edit these aliases
#ifdef W1
  const float* W1_ptr = W1;
  const float* B1_ptr = B1;
  const float* W2_ptr = W2;
  const float* B2_ptr = B2;
  const int HIDDEN1 = H1_DIM; // you must ensure generated header defines H1_DIM
#else
  const float* W1_ptr = W1_example;
  const float* B1_ptr = B1_example;
  const float* W2_ptr = W2_example;
  const float* B2_ptr = B2_example;
  const int HIDDEN1 = 8;
#endif

// Simple MLP inference: input 3 floats (normalized 0..1) -> hidden ReLU -> outputs 4 logits -> argmax
int mlp_predict(const float in0, const float in1, const float in2) {
  // hidden = ReLU(W1 * in + B1)
  float hidden[16]; // support up to 16 hidden neurons (ensure HIDDEN1<=16)
  for (int i = 0; i < HIDDEN1; ++i) {
    float s = B1_ptr[i];
    // W1 is row-major: W1[i*3 + j]
    s += W1_ptr[i*3 + 0] * in0;
    s += W1_ptr[i*3 + 1] * in1;
    s += W1_ptr[i*3 + 2] * in2;
    hidden[i] = (s > 0.0f) ? s : 0.0f;
  }
  float logits[4];
  for (int k = 0; k < 4; ++k) {
    float s = B2_ptr[k];
    // W2 is row-major: W2[k*HIDDEN1 + i]
    for (int i = 0; i < HIDDEN1; ++i) s += W2_ptr[k*HIDDEN1 + i] * hidden[i];
    logits[k] = s;
  }
  int best = 0;
  float bestv = logits[0];
  for (int i = 1; i < 4; ++i) if (logits[i] > bestv) { bestv = logits[i]; best = i; }
  return best;
}

// Ultrasonic helper
unsigned int readUltrasonicOnce() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long duration = pulseIn(ECHO, HIGH, 30000UL);
  if (duration == 0) return 999;
  unsigned int dist = (unsigned int)((duration * 0.034) / 2.0 + 0.5);
  return dist;
}
unsigned int readUltrasonicAvg(int samples=3) {
  long sum = 0; int valid = 0;
  for (int i=0;i<samples;i++){
    unsigned int d = readUltrasonicOnce();
    if (d < 999) { sum += d; valid++; }
    delay(10);
  }
  if (valid==0) return 999;
  return sum/valid;
}

void setup() {
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  pinMode(FRONT_LED, OUTPUT); pinMode(BACK_LED, OUTPUT);
  panServo.attach(SERVO_PIN);
  panServo.write(SERVO_CENTER);
  Serial.begin(115200);
  motorsStop();
  Serial.println("ANNie firmware ready");
}

void loop() {
  unsigned int front = readUltrasonicAvg();
  Serial.print("Front: "); Serial.println(front);
  if (front <= CRITICAL_DISTANCE) {
    Serial.println("CRITICAL STOP");
    motorsStop();
    delay(200);
    return;
  }
  if (front > SAFE_DISTANCE) {
    motorsForward(200);
    delay(80);
    return;
  }

  // Decision cycle: backup, scan, run ANN
  motorsStop(); delay(80);
  motorsBackward(180); delay(320); motorsStop(); delay(120);

  panServo.write(SERVO_LEFT); delay(300);
  unsigned int leftDist = readUltrasonicAvg();
  delay(60);

  panServo.write(SERVO_RIGHT); delay(300);
  unsigned int rightDist = readUltrasonicAvg();
  delay(60);

  panServo.write(SERVO_CENTER); delay(300);
  unsigned int frontFresh = readUltrasonicAvg();

  // Normalize to 0..1 (same as training generator)
  float in0 = (frontFresh >= 100 || frontFresh==999) ? 1.0f : (frontFresh / 100.0f);
  float in1 = (leftDist >= 100 || leftDist==999) ? 1.0f : (leftDist / 100.0f);
  float in2 = (rightDist >= 100 || rightDist==999) ? 1.0f : (rightDist / 100.0f);

  int action = mlp_predict(in0, in1, in2);
  Serial.print("ANN action: "); Serial.println(action);

  // safety override
  if (action == 0 && frontFresh <= SAFE_DISTANCE) {
    Serial.println("ANN FORWARD refused (blocked)");
    action = 3; // STOP fallback
  }

  // execute action briefly
  switch (action) {
    case 0: motorsForward(200); break;
    case 1: motorsTurnLeft(200); break;
    case 2: motorsTurnRight(200); break;
    case 3: motorsStop(); break;
  }
  delay(450);
  motorsStop();
  delay(120);
}