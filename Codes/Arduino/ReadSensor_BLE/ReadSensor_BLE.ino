#include <bluefruit.h>
#include <Adafruit_LittleFS.h>
#include <InternalFileSystem.h>
#include "Adafruit_MLX90393.h"

// Total three MLX90393 sensors of the sensor array for the mini car
#define num 3

Adafruit_MLX90393 sensor[num];
// CS pins
int CS[num] = {2, 3, 4};

// BLE Service
BLEDfu  bledfu;  // OTA DFU service
BLEDis  bledis;  // device information
BLEUart bleuart; // uart over ble
BLEBas  blebas;  // battery

float data_array[num*3];   // The array to hold the data

void setup()
{
  dwt_enable();         // For more accurate micros() on Feather
  Serial.begin(115200);
#if CFG_DEBUG
  // Blocking wait for connection when debug mode is enabled via IDE
  while ( !Serial ) yield();
#endif
  //********************BLE SETUP*******************//
  Serial.println("Bluefruit52 BLEUART Example");
  Serial.println("---------------------------\n");

  // Setup the BLE LED to be enabled on CONNECT
  // Note: This is actually the default behavior, but provided
  // here in case you want to control this LED manually via PIN 19
  Bluefruit.autoConnLed(true);

  // Config the peripheral connection with maximum bandwidth 
  // more SRAM required by SoftDevice
  // Note: All config***() function must be called before begin()
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);

  Bluefruit.begin();
  Bluefruit.setTxPower(4);    // Check bluefruit.h for supported values
  Bluefruit.setName("Bluefruit52");
  //Bluefruit.setName(getMcuUniqueID()); // useful testing with multiple central connections
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

  // To be consistent OTA DFU should be added first if it exists
  bledfu.begin();

  // Configure and Start Device Information Service
  bledis.setManufacturer("Adafruit Industries");
  bledis.setModel("Bluefruit Feather52");
  bledis.begin();

  // Configure and Start BLE Uart Service
  bleuart.begin();

  // Start BLE Battery Service
  blebas.begin();
  blebas.write(100);

  // Set up and start advertising
  startAdv();

  //*****************SENSOR SETUP*****************//
  pinMode(LED_BUILTIN, OUTPUT);     // Indicator of whether the sensors are all found
  digitalWrite(LED_BUILTIN, LOW);
  delayMicroseconds(2);
  for(int i = 0; i < num; ++i){
    sensor[i] = Adafruit_MLX90393();
    while (! sensor[i].begin_SPI(CS[i])) {
      Serial.print("No sensor ");
      Serial.print(i+1);
      Serial.println(" found ... check your wiring?");
      delayMicroseconds(500);
    } 
    Serial.print("Sensor ");
    Serial.print(i+1);
    Serial.println(" found!");
    // OSR: 3, Filter: 5, sapling rate: 17Hz
    while(!sensor[i].setOversampling(MLX90393_OSR_3)){
      Serial.print("Sensor ");
      Serial.print(i+1);
      Serial.println(" reset OSR!");
      delayMicroseconds(500);
    }
    delayMicroseconds(500);
    while(!sensor[i].setFilter(MLX90393_FILTER_5)){
      Serial.print("Sensor ");
      Serial.print(i+1);
      Serial.println(" reset filter!");
      delayMicroseconds(1000);
    }
  } 
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("Found all MLX90393 sensors");
}

void startAdv(void)
{
  // Advertising packet
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();

  // Include bleuart 128-bit uuid
  Bluefruit.Advertising.addService(bleuart);

  // Secondary Scan Response packet (optional)
  // Since there is no room for 'Name' in Advertising packet
  Bluefruit.ScanResponse.addName();
  
  /* Start Advertising
   * - Enable auto advertising if disconnected
   * - Interval:  fast mode = 20 ms, slow mode = 152.5 ms
   * - Timeout for fast mode is 30 seconds
   * - Start(timeout) with timeout = 0 will advertise forever (until connected)
   * 
   * For recommended advertising interval
   * https://developer.apple.com/library/content/qa/qa1931/_index.html   
   */
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds

}

void loop()
{ 
   Serial.println("###################");
   for(int i = 0; i < num; ++i){
      sensor[i].startSingleMeasurement();
      //delayMicroseconds(50);
   }
   delayMicroseconds(mlx90393_tconv[5][3]*1000+100); 
   for(int i = 0; i < num; ++i){
      if(!sensor[i].readMeasurement(&data_array[3*i], &data_array[3*i+1], &data_array[3*i+2])){
        Serial.print("Sensor ");
        Serial.print(i+1);
        Serial.println(" no data read!");
        digitalWrite(LED_BUILTIN, LOW);
      }
      delayMicroseconds(700);
//      Serial.print("Sensor ");
//      Serial.print(i+1);
//      Serial.print(data_array[3*i]);
//      Serial.print(" ");
//      Serial.print(data_array[3*i+1]);
//      Serial.print(" ");
//      Serial.print(data_array[3*i+2]);
    }

    // Send to PC using BLE
    bleuart.write((byte*)(data_array), num*3*4);
}

// callback invoked when central connects
void connect_callback(uint16_t conn_handle)
{
  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);

  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));

  Serial.print("Connected to ");
  Serial.println(central_name);
}

/**
 * Callback invoked when a connection is dropped
 * @param conn_handle connection where this event happens
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
  (void) conn_handle;
  (void) reason;

  Serial.println();
  Serial.print("Disconnected, reason = 0x"); Serial.println(reason, HEX);
}
