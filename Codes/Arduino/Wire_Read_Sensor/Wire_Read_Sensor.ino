#include "Adafruit_MLX90393.h"
#define num 9
Adafruit_MLX90393 sensor[num];
// int CS[] = {16, 15, 7, 11, 26, 25, 27, 30};
int CS[num] = {2, 3, 4, 5, 7, 9, 10, 11, 15};

// int CS[num] = {A0};
float data_array[num*3+1];
 
void setup() 
{
  // dwt_enable(); // For more accurate micros() on Feather
   Serial.begin(115200);
//  Serial.begin(921600);
  /* Wait for） s.erial on USB platforms. */
//  pinMode(LED_BUILTIN, OUTPUT); // Indicator of whether the sensors are all found 
//  digitalWrite(LED_BUILTIN, LOW);                     *1
  while (!Serial)
  {
    delayMicroseconds(10);
  }
  delayMicroseconds(1000);
  for (int i = 0; i < num; ++i)
  {
    sensor[i] = Adafruit_MLX90393();
    while (!sensor[i].begin_SPI(CS[i]))
    {
      Serial.print("No sensor ");
      Serial.print(i + 1);
      Serial.println(" found ... check your wiring?");
      delayMicroseconds(500);
    }
//    Serial.print("Sensor ");
//    Serial.print(i + 1);
//    Serial.println(" found!");
    while (!sensor[i].setOversampling(MLX90393_OSR_2))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" reset OSR!");
      // delayMicroseconds(500);
    }
    delayMicroseconds(500);
    while (!sensor[i].setFilter(MLX90393_FILTER_3))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" reset filter!");
      // delayMicroseconds(500);
    }
  }
//  digitalWrite(LED_BUILTIN, HIGH);
}

int cnt = 1;
void loop()
{ 
  String myString = "";
  int start_time = micros();
  for(int i = 0; i < num; ++i){
    sensor[i].startSingleMeasurement();
    }
    
    delayMicroseconds(mlx90393_tconv[3][2] * 1000+100);
    
    myString = "";
    for(int i = 0; i < num; ++i)
  {
    sensor[i].readMeasurement(&data_array[3*i], &data_array[3*i+1], &data_array[3*i+2]);
//    z[i+4] = z[i];
//    z[i+8] = z[i];
//    z[i+12] = z[i];
//    z[i+16] = z[i];
//    z[i+20] = z[i];
//    z[i+24] = z[i];
//    z[i+28] = z[i];
//    if (!sensor[i].readMeasurement(&data_array[3*i], &data_array[3*i+1], &data_array[3*i+2]))
//    {
////      Serial.print("Sensor ");
////      Serial.print(i+1);
////      Serial.println(" no data read!");
//      // digitalWrite(10, LOW);
//    }
     // sensor[i].readMeasurement(&data_array[3*i], &data_array[3*i+1], &data_array[3*i+2]);
//     myString.concat(data_array[3*i]);
//     myString += " ";
//     myString.concat(data_array[3*i+1]);
//     myString += " ";
//     myString.concat(data_array[3*i+2]);
//     myString += " ";
  }
    data_array[3*num] = micros() - start_time;
    Serial.write((byte*)(data_array), 4*(3*num+1));
    //int elapsed_time = micros() - start_time;
    // myString.concat(elapsed_time);
    // Serial.println(myString);
//    
}
