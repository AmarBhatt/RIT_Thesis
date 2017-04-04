/*
* Autonomous Obstacle Avoidance Vehicle
*
* Description: Vehicle that avoids obstacles and can navigate across a flat surface
*
* Author: Amar Bhatt
*/

// Define Motor Pins
const int motor1A = 9;
const int motor1B = 8;
const int motor2A = 7;
const int motor2B = 6;

//Define Enable Pins
const int motorE1 = 2;
const int motorE2 = 3;

//Define Speed
const int SPEED = 166;
const int TURN_SPEED = 166;
int currentSpeed = SPEED;

char inst;

// Initialize pins  
void setup(){
  Serial.begin(9600);
  
  //pinMode(motor1A, OUTPUT);
  //pinMode(motor1B, OUTPUT);
  
  //pinMode(motor2A, OUTPUT);
  //pinMode(motor2B, OUTPUT);

  //digitalWrite(motor1A, LOW);
  //digitalWrite(motor1B, LOW);
  //digitalWrite(motor2A, LOW);
  //digitalWrite(motor2B, LOW);

  analogWrite(motor1A, LOW);
  analogWrite(motor1B, LOW);
  analogWrite(motor2A, LOW);
  analogWrite(motor2B, LOW);
  
  Serial.println("Start Motors");
}

void loop(){
  
  while (Serial.available()){
  inst = Serial.read();
  switch (inst) {
    case 'F':
       setDirection(0);
       go();
       break;
    case  'B':
        setDirection(1);
        go();
        break;
    case  'L':
        setDirection(2);
        go();
        break;
    case 'R':
        setDirection(3);
        go();
        break;
    case 'S':
        brake();
        break;
    default:
        break;
  }
}
}

void go() {
  analogWrite(motorE1,HIGH);
  analogWrite(motorE2,HIGH);
  //delay(100);
  //brake();
}

void brake() {
  analogWrite(motorE1,0);
  analogWrite(motorE2,0);
  analogWrite(motor1A, LOW);
  analogWrite(motor1B, LOW);
  analogWrite(motor2A, LOW);
  analogWrite(motor2B, LOW);
}

void setDirection (int dir) {

  switch(dir){
    case 0: //Forward
      analogWrite(motor1B, LOW);
      analogWrite(motor2B, LOW);
      analogWrite(motor1A, currentSpeed);
      analogWrite(motor2A, currentSpeed);
      currentSpeed = SPEED;
      break;
    case 1: //Reverse
      analogWrite(motor1A, LOW);
      analogWrite(motor2A, LOW);
      analogWrite(motor1B, currentSpeed);
      analogWrite(motor2B, currentSpeed);
      break;
    case 2: //Turn left
      analogWrite(motor1A, currentSpeed);
      analogWrite(motor2B, currentSpeed);
      analogWrite(motor1B, LOW);
      analogWrite(motor2A, LOW); 
      currentSpeed = TURN_SPEED;
      break;
    default: //Turn right
      analogWrite(motor1B, currentSpeed);
      analogWrite(motor2A, currentSpeed);
      analogWrite(motor1A, LOW);
      analogWrite(motor2B, LOW);
      currentSpeed = TURN_SPEED;
      break;
  }       
}
