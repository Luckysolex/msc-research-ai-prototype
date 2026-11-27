/*
- On-Device Malaria Cell Classifier
- Hardware: Seeed Studio XIAO ESP32-S3 Sense
- Framework: TensorFlowLite_ESP32
- Author: Lucky Ayinwugyefun Solomon
- MSc Embedded AI — University of Abuja (Erasmus+ Co-funded)
- Hardware support: Seeed Studio, Shenzhen
*
- Description:
- Loads a fully INT8-quantized CNN trained on the NIH Malaria Cell
- Image Dataset. Classifies cells as Parasitized or Uninfected.
- Current stage: prototype with camera frame data.
*/

#include <Arduino.h>
#include <math.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "TensorFlowLite_ESP32.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "malaria_model.h"

// XIAO ESP32-S3 Sense camera pins (ESP32 Arduino core 2.x camera API).
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

constexpr int MODEL_WIDTH = 64;
constexpr int MODEL_HEIGHT = 64;
constexpr int MODEL_CHANNELS = 3;
constexpr size_t TENSOR_ARENA_SIZE = 160 * 1024;

// The generated xxd header is expected to define:
//   unsigned char malaria_quant_model_tflite[]
//   unsigned int malaria_quant_model_tflite_len
const unsigned char *g_model_data = malaria_quant_model_tflite;
const size_t g_model_length = malaria_quant_model_tflite_len;

uint8_t *tensor_arena = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
const tflite::Model *model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

bool cameraReady = false;

static int clampInt(int value, int low, int high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

static int8_t quantizePixel(uint8_t pixel, float scale, int zero_point) {
  // Normalize 0..255 pixel float to 0.0..1.0
  const float normalized = static_cast<float>(pixel) / 255.0f;
  
  // Quantize to int8: (normalized / scale) + zero_point
  int32_t quantized = static_cast<int32_t>(roundf(normalized / scale)) + zero_point;
  
  // Clamp to valid int8 range [-128, 127]
  return static_cast<int8_t>(clampInt(quantized, -128, 127));
}


static bool preprocessFrame(const camera_fb_t *fb) {
  if (!fb || !fb->buf || !input) return false;
  if (fb->format != PIXFORMAT_RGB565) return false;

  const int source_width = fb->width;
  const int source_height = fb->height;

  if (source_width <= 0 || source_height <= 0) return false;

  const size_t required_bytes =
      static_cast<size_t>(source_width) * source_height * 2;

  if (fb->len < required_bytes) return false;

Serial.println("\n--- CAMERA DIAGNOSTICS ---");

Serial.printf("Frame width: %d\n", fb->width);
Serial.printf("Frame height: %d\n", fb->height);
Serial.printf("Frame length: %u\n", (unsigned int)fb->len);
Serial.printf("Pixel format: %d\n", fb->format);

Serial.printf("First two bytes: 0x%02X 0x%02X\n",
              fb->buf[0], fb->buf[1]);

uint16_t pixel_big_endian =
    ((uint16_t)fb->buf[0] << 8) | fb->buf[1];

uint16_t pixel_little_endian =
    ((uint16_t)fb->buf[1] << 8) | fb->buf[0];

Serial.printf("Big-endian RGB565: 0x%04X\n", pixel_big_endian);
Serial.printf("Little-endian RGB565: 0x%04X\n", pixel_little_endian);

  const float input_scale = input->params.scale;
  const int zero_point = input->params.zero_point;

  if (input_scale <= 0.0f) return false;

  int index = 0;

  for (int y = 0; y < MODEL_HEIGHT; ++y) {
    // Direct resize: map each output pixel to the full source image.
    const int sy = (y * source_height) / MODEL_HEIGHT;

    for (int x = 0; x < MODEL_WIDTH; ++x) {
      const int sx = (x * source_width) / MODEL_WIDTH;

      const size_t byte_index =
          (static_cast<size_t>(sy) * source_width + sx) * 2;

      // Interpret RGB565 bytes as big-endian.
      // Verify this byte order against the camera output if predictions
      // remain poor.
      const uint8_t high_byte = fb->buf[byte_index];
      const uint8_t low_byte = fb->buf[byte_index + 1];

      const uint16_t pixel =
          (static_cast<uint16_t>(high_byte) << 8) | low_byte;

      const uint8_t r5 = (pixel >> 11) & 0x1F;
      const uint8_t g6 = (pixel >> 5) & 0x3F;
      const uint8_t b5 = pixel & 0x1F;

      const uint8_t red   = (r5 * 255 + 15) / 31;
      const uint8_t green = (g6 * 255 + 31) / 63;
      const uint8_t blue  = (b5 * 255 + 15) / 31;

      input->data.int8[index++] =
          quantizePixel(red, input_scale, zero_point);
      input->data.int8[index++] =
          quantizePixel(green, input_scale, zero_point);
      input->data.int8[index++] =
          quantizePixel(blue, input_scale, zero_point);
    }
  }

  return true;
}

static bool initializeCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;  // 160x120
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_framesize(sensor, FRAMESIZE_QQVGA);
  }
  return true;
}


static void testCameraJPEG() {
  sensor_t *sensor = esp_camera_sensor_get();

  if (!sensor) {
    Serial.println("ERROR: Camera sensor unavailable.");
    return;
  }

  Serial.println("\n--- JPEG CAMERA TEST ---");

  // Temporarily switch the sensor to JPEG.
  sensor->set_pixformat(sensor, PIXFORMAT_JPEG);
  delay(300);

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("ERROR: JPEG capture failed.");
  } else {
    Serial.printf("JPEG frame size: %u bytes\n",
                  (unsigned int)fb->len);

    Serial.printf("JPEG dimensions: %d x %d\n",
                  fb->width, fb->height);

    Serial.printf("JPEG format: %d\n", fb->format);

    esp_camera_fb_return(fb);
  }

  // Restore the original RGB565 format.
  sensor->set_pixformat(sensor, PIXFORMAT_RGB565);
  delay(300);

  Serial.println("Camera restored to RGB565.");
}


static bool initializeModel() {
  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not detected. Enable OPI PSRAM in Tools.");
    return false;
  }

  tensor_arena = static_cast<uint8_t *>(
      heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!tensor_arena) {
    Serial.println("ERROR: Could not allocate tensor arena in PSRAM.");
    return false;
  }

  model = tflite::GetModel(g_model_data);
  
  if (!model) {
    Serial.println("ERROR: Could not read TFLite model.");
    return false;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR: Model schema %d does not match runtime schema %d.\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  interpreter = new (std::nothrow) tflite::MicroInterpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  if (!interpreter) {
    Serial.println("ERROR: Could not create MicroInterpreter.");
    return false;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed. Tensor arena may be too small.");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (!input || !output) {
    Serial.println("ERROR: Model input or output tensor is missing.");
    return false;
  }

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Serial.println("ERROR: Expected int8 input and output tensors.");
    return false;
  }

  if (input->dims->size != 4 ||
      input->dims->data[1] != MODEL_HEIGHT ||
      input->dims->data[2] != MODEL_WIDTH ||
      input->dims->data[3] != MODEL_CHANNELS) {
    Serial.println("ERROR: Model input is not [1,64,64,3].");
    return false;
  }

  Serial.printf("Model loaded: %u bytes\n", static_cast<unsigned>(g_model_length));
  Serial.printf("Input quantization: scale=%.8f zero_point=%d\n",
                input->params.scale, input->params.zero_point);
  Serial.printf("Output quantization: scale=%.8f zero_point=%d\n",
                output->params.scale, output->params.zero_point);
  return true;
}

static void runInference() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ERROR: Camera frame capture failed! Retrying...");
    delay(500);
    return;
  }

  const bool preprocessed = preprocessFrame(fb);
  esp_camera_fb_return(fb); // Always release the framebuffer back to the camera driver

  if (!preprocessed) {
    Serial.println("ERROR: Frame preprocessing failed.");
    return;
  }

  const uint32_t start_us = micros();

// Run model inference
if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Model inference failed.");
    return;
}

const uint32_t elapsed_us = micros() - start_us;

Serial.printf(
    "Raw output INT8: %d\n",
    static_cast<int>(output->data.int8[0])
);

Serial.printf(
    "Input tensor: scale=%.8f, zero_point=%d\n",
    input->params.scale,
    input->params.zero_point
);

Serial.printf(
  "Inference time: %lu us (%.3f ms)\n",
      static_cast<unsigned long>(elapsed_us),
      elapsed_us / 1000.0f
);

  const float score =
      (static_cast<int>(output->data.int8[0]) - output->params.zero_point) *
      output->params.scale;

  // Verified training convention: class 0 = Parasitized, class 1 = Uninfected.
  const char *label = (score >= 0.5f) ? "Uninfected" : "Parasitized";

  Serial.println("--------------------------------");
  Serial.printf("Prediction: %s\n", label);
  Serial.printf("Class-1 (Uninfected) score: %.4f\n", score);
  Serial.printf("Inference time: %lu us (%.3f ms)\n",
              static_cast<unsigned long>(elapsed_us),
              elapsed_us / 1000.0f);
  Serial.println("Research prototype only; not for clinical diagnosis.");
}


void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("\n--- XIAO ESP32-S3 Malaria Classifier ---");

  if (!initializeModel()) {
    Serial.println("CRITICAL ERROR: Model Initialization Failed!");
    while (true) {
      delay(1000);
    }
  }

  Serial.printf(
      "Input tensor dims: %d x %d x %d x %d\n",
      input->dims->data[0],
      input->dims->data[1],
      input->dims->data[2],
      input->dims->data[3]
  );

  if (!initializeCamera()) {
    Serial.println("CRITICAL ERROR: Camera Initialization Failed!");
    while (true) {
      delay(1000);
    }
  }

  cameraReady = true;
  Serial.println("Camera initialized successfully!");
  Serial.println("Starting camera inference...");
}
void loop() {
  if (cameraReady) {
    runInference();
  }
  delay(1000);
}