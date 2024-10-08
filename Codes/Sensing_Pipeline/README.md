# Sensing Pipeline

## Description
- The `C_Implementation` folder contains a real-time sensing algorithm for detecting Polaris tags implemented in C. 
This code is optimized for low-power devices such as ESP32 chips due to its speed and resource efficiency.

- The `Python_Implementation` folder contains the detection algorithm implemented in Python, which can accelerate algorithm implementation and enhances cross-platform compatibility. It includes:
    - `sensing_real_time.py`: 
    the real-time sensing algorithm for detecting a Polaris tag.
    - `/Utils` folder: 
    contains required utility functions for the sensing pipeline. Specifically, to accelerate the execution of the Python code, we implemented the DDTW algorithm and localization algorithms in C, and embedded them into the Python code as  `.so` (shared object) extension files.
        - `ddtw.c`: C implementation of DDTW algorithm for the polarity orientation.
        - `localize.c`: C implementation of the localization algorithm.
        - `ddtw.so` and `localize.so`: precompiled shared object extension files. 
        To build the `.so` files yourself, use the following command:
            ```
            gcc -shared -o ddtw.so -fPIC ddtw.c -lm 
            ```
    - `template_36_5_80.txt`: 
    this template is generated by synthesizing magnetometer data during the offline stage using Magpylib. 
    It is used to determine the relative orientation of the sensor array with respect to the magnet by DDTW matching process. 
    For more details, please refer to Sec. 6.2.2 of the paper.
    - `requirements.txt`: lists the required Python libraries for the sensing pipeline. 
    You can install each library using the command pip install [library_name]

- The `Simulate_Engine` contains the Python code for generating synthetic magnetic field data templates (e.g., `template_36_5_80.txt`), based on Magpylib. 
The generated templates are used for the magnet polarity orientation matching algorithm. 
For details, please refer to Sec. 6.2.2.

## Note
This directory contains the sensing pipeline for real-time detection, which requires hardwares, e.g., the sensor array and a robot car.
For a quick start with our Polaris system, we’ve provided an offline version of the sensing algorithm framework. 
For more information, please refer to the related [README](../../Evaluation/README.md) file.