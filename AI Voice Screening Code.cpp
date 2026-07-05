#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <driver/i2s.h>
#include <SD.h>
#include <SPI.h>
#include "esp_task_wdt.h"
#include <your_model_inference.h>

#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32
#define BUTTON_PIN 26
#define SD_CS 5
#define OLED_SDA 21
#define OLED_SCL 22
#define SPEAKER_PIN 27

#define SAMPLE_RATE 16000
#define SAMPLE_BITS 16
#define RECORD_TIME 5
#define BUFFER_SIZE 512
#define DMA_BUF_COUNT 8
#define DMA_BUF_LEN 1024

Adafruit_SH1106G display(128, 64, &Wire, -1);

int16_t audioBuffer[BUFFER_SIZE];
float inferenceBuffer[SAMPLE_RATE * RECORD_TIME];
int inferenceBufferIndex = 0;
bool isRecording = false;
bool buttonPressed = false;
unsigned long recordStartTime = 0;

String diagnosisResult = "Ready";
float confidence = 0.0;

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = DMA_BUF_COUNT,
    .dma_buf_len = DMA_BUF_LEN,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

void setupDisplay() {
  Wire.begin(OLED_SDA, OLED_SCL);
  display.begin(0x3C, true);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 0);
  display.println("Voice Screening");
  display.println("Device Ready");
  display.println("");
  display.println("Press button to");
  display.println("start recording");
  display.display();
}

void updateDisplay(String status, String result = "", float conf = 0.0) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.println("Voice Screening");
  display.println("----------------");
  display.println(status);
  
  if (result != "") {
    display.println("");
    display.setTextSize(1);
    display.println("Result:");
    display.setTextSize(2);
    display.println(result);
    display.setTextSize(1);
    display.print("Conf: ");
    display.print(conf, 1);
    display.println("%");
  }
  display.display();
}

void setupSDCard() {
  if (!SD.begin(SD_CS)) {
    updateDisplay("SD Card Error", "", 0);
    return;
  }
}

void IRAM_ATTR buttonISR() {
  buttonPressed = true;
}

void recordAudio() {
  updateDisplay("Recording...", "", 0);
  isRecording = true;
  recordStartTime = millis();
  inferenceBufferIndex = 0;

  size_t bytesRead = 0;
  int totalSamples = SAMPLE_RATE * RECORD_TIME;
  
  while (inferenceBufferIndex < totalSamples && isRecording) {
    i2s_read(I2S_NUM_0, &audioBuffer, BUFFER_SIZE * sizeof(int16_t), &bytesRead, portMAX_DELAY);
    int samplesRead = bytesRead / sizeof(int16_t);

    for (int i = 0; i < samplesRead && inferenceBufferIndex < totalSamples; i++) {
      inferenceBuffer[inferenceBufferIndex++] = (float)audioBuffer[i] / 32768.0f;
    }

    if (millis() - recordStartTime > (RECORD_TIME * 1000)) {
      isRecording = false;
    }
  }
  updateDisplay("Processing...", "", 0);
}

static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
  for (size_t i = 0; i < length; i++) out_ptr[i] = inferenceBuffer[offset + i];
  return 0;
}

void runInference() {
  signal_t signal;
  signal.total_length = inferenceBufferIndex;
  signal.get_data = &get_signal_data;

  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

  if (res != EI_IMPULSE_OK) {
    updateDisplay("Error", "Inference failed", 0);
    return;
  }

  float maxConfidence = 0.0;
  String prediction = "Unknown";

  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > maxConfidence) {
      maxConfidence = result.classification[i].value;
      prediction = String(result.classification[i].label);
    }
  }

  confidence = maxConfidence * 100.0;
  diagnosisResult = prediction;
  updateDisplay("Complete!", diagnosisResult, confidence);

  speakResult(diagnosisResult);
  saveResultToSD(diagnosisResult, confidence);
}

void saveResultToSD(String result, float conf) {
  File dataFile = SD.open("/results.txt", FILE_APPEND);
  
  if (dataFile) {
    String timestamp = String(millis());
    dataFile.print(timestamp);
    dataFile.print(",");
    dataFile.print(result);
    dataFile.print(",");
    dataFile.println(conf);
    dataFile.close();
  }
}

void speakResult(String result) {
  if (result == "Healthy") {
    playTone(800, 200);
    delay(100);
    playTone(1000, 200);
  } else if (result == "Parkinsons" || result == "Parkinson") {
    playTone(600, 300);
    delay(100);
    playTone(500, 300);
    delay(100);
    playTone(400, 300);
  } else if (result == "ALS") {
    playTone(500, 400);
    delay(100);
    playTone(400, 400);
  }
}

void playTone(int frequency, int duration) {
  ledcSetup(0, frequency, 8);
  ledcAttachPin(SPEAKER_PIN, 0);
  ledcWrite(0, 128);
  delay(duration);
  ledcWrite(0, 0);
}

void setup() {
  Serial.begin(115200);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

  setupDisplay();
  setupI2S();
  setupSDCard();

  if (run_classifier_init() != EI_IMPULSE_OK) {
    updateDisplay("Error", "AI Init Failed", 0);
    return;
  }

  updateDisplay("Ready", "Press to start", 0);
}

void loop() {
  if (buttonPressed) {
    buttonPressed = false;
    delay(200);
    recordAudio();
    runInference();
    delay(3000);
    updateDisplay("Ready", "Press to start", 0);
  }
}
