# On-Device Malaria Cell Classifier

**A TinyML research prototype for malaria cell classification using the ESP32-S3 and TensorFlow Lite Micro.**

## Project Overview

This project explores the deployment of a lightweight convolutional neural network (CNN) for binary malaria cell classification on resource-constrained embedded hardware.

The model classifies microscopic red blood cell images into two categories:

- **Parasitized:** Cells containing malaria parasites.
- **Uninfected:** Cells without malaria parasites.

The research investigates how machine learning models can be quantized and deployed for on-device inference without depending on cloud connectivity.

## Hardware and Technologies

| Component | Technology |
|---|---|
| Microcontroller | Seeed Studio XIAO ESP32-S3 Sense |
| Camera | OV2640 |
| Model architecture | Convolutional Neural Network |
| ML framework | TensorFlow / TensorFlow Lite |
| Embedded inference | TensorFlow Lite Micro |
| Programming language | Python, C++ |
| Model quantization | Full INT8 |

## Model Specifications

- Input dimensions: 64 × 64 × 3
- Input type: INT8
- Output type: INT8
- Number of classes: 2
- Quantized model size: 79,784 bytes

## Experimental Results

The fully quantized model achieved the following results on a held-out test dataset:

| Metric | Result |
|---|---:|
| Test images | 2,759 |
| Accuracy | 96.12% |
| Parasitized recall | 95% |
| Uninfected recall | 97% |

These results represent offline evaluation of the quantized model, not independently established hardware or clinical diagnostic accuracy.

## Embedded Deployment

The model has been integrated into an ESP32-S3 prototype using TensorFlow Lite Micro.

The prototype has demonstrated:
- Successful model loading on the microcontroller.
- OV2640 camera initialization and frame capture.
- On-device model inference.
- Classification output through the serial monitor.

A measured inference time of approximately 1.83 seconds was observed in an earlier prototype run. Further hardware-level validation is ongoing.

## Repository Structure

```text
msc-research-ai-prototype/
│
├── notebooks/
│   └── malaria_model_training.ipynb
│
├── firmware/
│   └── malaria_classifier/
│       ├── README.md
│       └── [Arduino source files]
│
├── docs/
│   └── DATASET.md
│
├── results/
│   └── model_evaluation.md
│
├── README.md
└── THESIS_LOG.md
```

## Research Objectives

1. Develop a CNN for malaria cell image classification.
2. Evaluate model performance using a held-out test dataset.
3. Apply full INT8 quantization to reduce deployment requirements.
4. Deploy the quantized model on an ESP32-S3 microcontroller.
5. Investigate the feasibility of low-cost, on-device malaria image classification.

## Current Development Status

The model training, quantization and initial embedded deployment have been completed. Camera image preprocessing and hardware-level validation remain ongoing.

## Dataset

The project uses the NIH Malaria Cell Image Dataset. Refer to [`docs/DATASET.md`](docs/DATASET.md) for dataset documentation.

## Disclaimer

This is an academic research prototype. It is not a clinically validated medical device and must not be used to diagnose or guide the treatment of patients.

## Research Context

MSc Embedded Artificial Intelligence  
University of Abuja

Hardware support: Seeed Studio

**Research focus:** TinyML, Edge AI and embedded machine learning for resource-constrained environments.
