# Firmware – On-Device Malaria Cell Classifier

This directory contains the Arduino firmware for deploying the quantized malaria cell classification model on the Seeed Studio XIAO ESP32-S3 Sense.

## Hardware Requirements
- Seeed Studio XIAO ESP32-S3 Sense
- OV2640 camera module

## Software Requirements
- Arduino IDE
- ESP32 board package by Espressif Systems — version **2.0.17**
- `TensorFlowLite_ESP32 library — version 1.0.0` library
- ESP32 camera support provided by the ESP32 Arduino core

## Firmware Files
- `malaria_detection.ino` – Main firmware for camera initialization, image preprocessing, and on-device inference.
- `malaria_model.h` – C header containing the quantized model data.

## Model Configuration
- Input shape: `[1, 64, 64, 3]`
- Input type: INT8
- Output shape: `[1, 1]`
- Model size: 79,784 bytes
- Classes: `Parasitized` and `Uninfected`

## Deployment
1. Open `malaria_detection.ino` in Arduino IDE.
2. Ensure the ESP32 Arduino core and required TensorFlow Lite library are installed.
3. Select the Seeed Studio XIAO ESP32S3 board.
4. Connect the board and select its serial port.
5. Compile and upload the firmware.
6. Open the Serial Monitor at **115200 baud** to observe camera initialization and inference output.

**Note:** The model header is included directly by the firmware. The standalone `.tflite` file in the repository is provided for inspection and reproducibility.

## Research Disclaimer
This is an experimental research prototype. It is not clinically validated and must not be used for medical diagnosis.
