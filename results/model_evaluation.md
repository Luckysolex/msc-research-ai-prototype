# Model Evaluation Results

## On-Device Malaria Cell Classifier

### 1. Test Dataset Performance

The fully INT8-quantized TensorFlow Lite model was evaluated on a held-out test dataset containing 2,759 images.

| Metric | Result |
|---|---:|
| Test accuracy | 96.12% |
| Parasitized precision | 97% |
| Parasitized recall | 95% |
| Parasitized F1-score | 96% |
| Uninfected precision | 95% |
| Uninfected recall | 97% |
| Uninfected F1-score | 96% |

### 2. Confusion Matrix

Class order: Parasitized, Uninfected.

| Actual / Predicted | Parasitized | Uninfected |
|---|---:|---:|
| Parasitized | 1,315 | 70 |
| Uninfected | 37 | 1,337 |

Correct predictions: 2,652  
Incorrect predictions: 107  
Total test images: 2,759

### 3. Quantized Model Specifications

- Input shape: 1 × 64 × 64 × 3
- Input data type: INT8
- Output shape: 1 × 1
- Output data type: INT8
- Input quantization scale: 0.0038487303536385298
- Input zero point: -128
- Output quantization scale: 0.00390625
- Output zero point: -128
- Model size: 79,784 bytes

### 4. Hardware Deployment

Target hardware: Seeed Studio XIAO ESP32-S3 Sense.

Camera: OV2640.

Inference framework: TensorFlow Lite Micro.

A working prototype has demonstrated camera capture and on-device inference.

### 5. Important Limitations

The 96.12% accuracy represents performance on the held-out dataset using the quantized model. It does not represent independently measured ESP32 system accuracy or clinical diagnostic accuracy.

Further work is required to validate camera image preprocessing, hardware-level classification performance and robustness under real-world conditions.

**Research prototype only. Not intended for clinical diagnosis.**
