#include "Adafruit_MLX90393.h"

// Total nine MLX90393 sensors for the sensor array
#define num 9
Adafruit_MLX90393 sensor[num];

// CS pins 
int CS[num] = {2, 3, 4, 5, 7, 9, 10, 11, 15};
//int CS[num] = {2, 3, 4};
float data_array[num*3];

void setup()
{
  // baud rate: 115200
  Serial.begin(115200);
  /* Wait for serial on USB platforms. */
  while (!Serial)
  {
    delayMicroseconds(10);
  }
  pinMode(LED_BUILTIN, OUTPUT);     // Indicator of whether the sensors are all found
  digitalWrite(LED_BUILTIN, LOW);
  delayMicroseconds(2);
  delayMicroseconds(1000);
  for (int i = 0; i < num; ++i)
  {
    sensor[i] = Adafruit_MLX90393();
    // Use SPI protocol
    while (!sensor[i].begin_SPI(CS[i]))
    {
      Serial.print("No sensor ");
      Serial.print(i + 1);
      Serial.println(" found ... check your wiring?");
      delayMicroseconds(500);
    }
    
    // OSR: 3, Filter: 3, sapling rate: 100Hz
    while (!sensor[i].setOversampling(MLX90393_OSR_3))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" reset OSR!");
    }
    delayMicroseconds(500);
    while (!sensor[i].setFilter(MLX90393_FILTER_3))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" reset filter!");
    }
    
  }
  digitalWrite(LED_BUILTIN, HIGH);
}


void loop()
{ 
  int start_time = micros();
  for(int i = 0; i < num; ++i)
  {
    sensor[i].startSingleMeasurement();
  }
  // time for converting data
  delayMicroseconds(mlx90393_tconv[3][3] * 1000+300);
  for(int i = 0; i < num; ++i)
  {
    sensor[i].readMeasurement(&data_array[3*i], &data_array[3*i+1], &data_array[3*i+2]);
  }
  // write bytes to PC
  Serial.write((byte*)(data_array), 4*(3*num)); 
}
