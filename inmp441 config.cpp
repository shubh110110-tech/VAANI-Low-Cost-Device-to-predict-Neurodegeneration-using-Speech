#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_WS 15
#define I2S_SD 32
#define I2S_SCK 14

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024

void startI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

void setup() {
  Serial.begin(115200);
  Serial.println("Type '1' to start recording...");
  startI2S();
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      Serial.println("Recording...");
      int32_t buffer[BUFFER_SIZE];
      size_t bytesRead;

      while (!Serial.available()) {
        i2s_read(I2S_NUM_0, buffer, BUFFER_SIZE * sizeof(int32_t), &bytesRead, portMAX_DELAY);

        for (int i = 0; i < BUFFER_SIZE; i++) {
          int16_t sample = buffer[i] >> 16;
          Serial.write((uint8_t*)&sample, 2);
        }
      }
      Serial.println("Stopped.");
    }
  }
}
