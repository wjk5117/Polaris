# Hardware Design of Polaris' Sensor Array

## Description
This directory contains the comprehensive PCB design for Polaris' sensor array, including the sensor bar for fiducial detection and a flashing board designed to program the MCU (MDBT42Q-512KV2). 

"SensorArray_RobotCar" refers to the PCB design of the sensor array for the robot car.
"SensorArray_MiniCar" referes to the PCB design of the sensor array for the mini car.
"Flashing_Module" refers to the flashing board for programming the MCU to obtain data readings of each magnetometers.

![plot](../Imgs/sensing_array.png)

## Hardware requirements
The hardware setup requires various components during the design and fabrication process. 
Below is a list of components we used for designing and manufacturing our prototype.

For the sensor array:
| Component  | Description | Quantity |
| ------------- | ------------- | ------------- |
|[Raytac MDBT42Q-512KV2](https://www.digikey.com/en/products/detail/raytac/MDBT42Q-512KV2/13677592) | Microcontroller Unit | 1 |
|[Melexis MLX90393](https://www.digikey.com/en/products/detail/melexis-technologies-nv/MLX90393SLW-ABA-011-RE/5031684) | Magnetometer | M |
|[Seiko Epson Q13FC13500004](https://www.lcsc.com/product-detail/Crystals_span-style-background-color-ff0-Seiko-span-span-style-background-color-ff0-Epson-span-Q13FC13500004_C32346.html)  | 32.768kHz Surface mount crystal | 1 |
| [Diodes Incorporated AP2112K-3.3TRG1](https://www.lcsc.com/product-detail/Voltage-Regulators-Linear-Low-Drop-Out-LDO-Regulators_Diodes-Incorporated-AP2112K-3-3TRG1_C51118.html)  | Voltage regulator  | 1 |
| [SHOU HAN FPC 0.5-8P HYH2.0](https://www.lcsc.com/product-detail/FFC-span-style-background-color-ff0-FPC-span-Flat-Flexible-Connector-Assemblies_SHOU-HAN-FPC-0-5-8P-HYH2-0_C6364658.html) |FPC (Flat Flexible) Connector | 1 |
|[C&K KMR231GLFS](https://www.lcsc.com/product-detail/Tactile-Switches_C-K-KMR231GLFS_C99271.html) | Tactile Switch | 1 |
| [Vishay Intertech SI2301CDS-T1-GE3](https://www.lcsc.com/product-detail/MOSFETs_Vishay-Intertech-SI2301CDS-T1-GE3_C10487.html) |  P-Channel SOT23 MOSFET | 1 |
| [KEXIN 1N4148WS](https://www.lcsc.com/product-detail/Others_KEXIN-1N4148WS_C369921.html) | Switching Diodes | 1 |
| [LRC LMBR120FT1G](https://www.lcsc.com/product-detail/Schottky-Diodes_LRC-LMBR120FT1G_C81143.html) | Schottky Diodes | 1 |
| xx | Connector header | 1 |
| Resistor | Chip resistors, 0402 | 1kΩ (2), 10kΩ (1), 100kΩ (2) |
| Capacitor | Ceramic Capacitors, 0402| 10uF (3), 1uF (1), 12pF (2), 0.1uF (M) |
| Inductor | Inductors, 0402| 10uH (1), 15nH (1) |
| LED1 | LED Indication, Red | 1 |
| LED2| LED Indication, Blue | 1 |

For the flashing module:
| Component  | Description | Quantity | 
| ------------- | ------------- |------------- |
| [Diodes Incorporated AP2112K-3.3TRG1](https://www.lcsc.com/product-detail/Voltage-Regulators-Linear-Low-Drop-Out-LDO-Regulators_Diodes-Incorporated-AP2112K-3-3TRG1_C51118.html)  | Voltage regulator  | 1 |
| [Vishay Intertech SI2301CDS-T1-GE3](https://www.lcsc.com/product-detail/MOSFETs_Vishay-Intertech-SI2301CDS-T1-GE3_C10487.html) |  P-Channel SOT23 MOSFET | 1 |
| [LRC LMBR120FT1G](https://www.lcsc.com/product-detail/Schottky-Diodes_LRC-LMBR120FT1G_C81143.html) | Schottky Diodes | 1 |
| [Microchip Tech MCP73831T-2ATI/OT](https://www.lcsc.com/product-detail/Battery-Management_Microchip-Tech-MCP73831T-2ATI-OT_C14879.html) | Battery Management, SOT-23-5 | 1 |
| [SKYWORKS/SILICON LABS CP2104-F03-GM](https://www.lcsc.com/product-detail/USB-Converters_SKYWORKS-SILICON-LABS-CP2104-F03-GM_C430013.html) | USB Converter, QFN-24-EP(4x4)| 1 |
| [TFM-105-12-S-D-A](https://www.digikey.com/en/products/detail/samtec-inc/TFM-105-12-S-D-A/66788380) | Connector Header Surface Mount 10 position  | 1 |
| [SHOU HAN MicroXNJ](https://www.lcsc.com/product-detail/USB-Connectors_SHOU-HAN-MicroXNJ_C404969.html) | Micro-B SMD USB Connectors | 1 |
| [SHOU HAN FPC 0.5-8P HYH2.0](https://www.lcsc.com/product-detail/FFC-span-style-background-color-ff0-FPC-span-Flat-Flexible-Connector-Assemblies_SHOU-HAN-FPC-0-5-8P-HYH2-0_C6364658.html) |FPC (Flat Flexible) Connector | 1 |
| Resistor | Chip resistors, 0603 | 1kΩ (3), 100kΩ (3) |
| Capacitor 0.1uF | Ceramic Capacitors, 0603|  2 |
| Capacitor 10uF | Ceramic Capacitors, 0805| 5 |
| LED1 | LED Indication, Orange | 1 |
| xx | Connector header | 1 |



For each module, we also provide an interactive HTML BOM file for reference.


## Design tool and library
We utilize [Altium Designer](https://www.altium.com/altium-designer) (version 20.0.13) to design PCB boards.
The component library of capacitive and resistance relies on the standard library.
The footprint of the MLX90393 magnetometer we used, can see https://www.snapeda.com/parts/MLX90393SLW-ABA-011-RE/Melexis%20Technologies/view-part/551380/?ref=search&t=MLX90393