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
const int SPEED = 255;
const int TURN_SPEED = 255;
int currentSpeed = SPEED;

char inst;

// Initialize pins  
void setup(){
  Serial.begin(9600);
  
  pinMode(motor1A, OUTPUT);
  pinMode(motor1B, OUTPUT);
  
  pinMode(motor2A, OUTPUT);
  pinMode(motor2B, OUTPUT);

  digitalWrite(motor1A, LOW);
  digitalWrite(motor1B, LOW);
  digitalWrite(motor2A, LOW);
  digitalWrite(motor2B, LOW);
  
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
  analogWrite(motorE1,currentSpeed);
  analogWrite(motorE2,currentSpeed);
  //delay(100);
  //brake();
}

void brake() {
  analogWrite(motorE1,0);
  analogWrite(motorE2,0);
  digitalWrite(motor1A, LOW);
  digitalWrite(motor1B, LOW);
  digitalWrite(motor2A, LOW);
  digitalWrite(motor2B, LOW);
}

void setDirection (int dir) {

  switch(dir){
    case 0: //Forward
      digitalWrite(motor1B, LOW);
      digitalWrite(motor2B, LOW);
      digitalWrite(motor1A, HIGH);
      digitalWrite(motor2A, HIGH);
      currentSpeed = SPEED;
      break;
    case 1: //Reverse
      digitalWrite(motor1A, LOW);
      digitalWrite(motor2A, LOW);
      digitalWrite(motor1B, HIGH);
      digitalWrite(motor2B, HIGH);
      break;
    case 2: //Turn left
      digitalWrite(motor1B, LOW);
      digitalWrite(motor2A, LOW);      
      digitalWrite(motor1A, HIGH);
      digitalWrite(motor2B, HIGH);
      currentSpeed = TURN_SPEED;
      break;
    default: //Turn right
      digitalWrite(motor1A, LOW);
      digitalWrite(motor2B, LOW);      
      digitalWrite(motor1B, HIGH);
      digitalWrite(motor2A, HIGH);
      currentSpeed = TURN_SPEED;
      break;
  }       
}
