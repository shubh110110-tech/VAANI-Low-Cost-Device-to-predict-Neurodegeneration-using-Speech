#include <Vaani_inferencing.h>
#include <driver/i2s.h>

#define I2S_WS   25
#define I2S_SD   35
#define I2S_SCK  26
#define I2S_PORT I2S_NUM_0

#define SAMPLE_BUFFER EI_CLASSIFIER_RAW_SAMPLE_COUNT
static int32_t samples[SAMPLE_BUFFER];
static signed short inferenceBuffer[SAMPLE_BUFFER];
static bool debug_nn = false;

void setupI2S() {
    const i2s_config_t config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = EI_CLASSIFIER_FREQUENCY,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
    Serial.println("✓ Microphone ready");
}

bool getAudioSamples() {
    size_t bytesRead;
    if (i2s_read(I2S_PORT, samples, sizeof(samples), &bytesRead, portMAX_DELAY) != ESP_OK) {
        return false;
    }
    int count = bytesRead / 4;
    for (int i = 0; i < count; i++) {
        inferenceBuffer[i] = samples[i] >> 14;
    }
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n========================================");
    Serial.println("   VAANI NEUROLOGICAL SCREENING");
    Serial.println("========================================");
    Serial.println("Model: Parkinsons / Healthy / ALS");
    Serial.println("Speak AH sound continuously...\n");

    setupI2S();
}

void loop() {
    if (!getAudioSamples()) return;

    // Volume check
    long sum = 0;
    for (int i = 0; i < SAMPLE_BUFFER; i++)
        sum += abs(inferenceBuffer[i]);
    int volume = sum / SAMPLE_BUFFER;

    // Run classifier
    signal_t signal;
    signal.total_length = SAMPLE_BUFFER;
    signal.get_data = [](size_t offset, size_t length, float *out_ptr) {
        numpy::int16_to_float(&inferenceBuffer[offset], out_ptr, length);
        return 0;
    };

    ei_impulse_result_t result;
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        Serial.println("Classifier error!");
        return;
    }

    // Find best prediction
    float best = 0;
    const char* label = "";
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best) {
            best = result.classification[i].value;
            label = result.classification[i].label;
        }
    }

    // Print full result
    Serial.println("\n========================================");
    Serial.println("         VAANI DIAGNOSIS RESULT");
    Serial.println("========================================");
    Serial.printf("  Volume Level : %d\n", volume);
    Serial.println("  ----------------------------------------");
    Serial.println("  Risk Scores:");
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float pct = result.classification[i].value * 100.0f;
        Serial.printf("  %-15s: %6.2f%%\n",
            result.classification[i].label, pct);
    }
    Serial.println("  ----------------------------------------");
    Serial.printf("  Diagnosis    : %s\n", label);
    Serial.printf("  Confidence   : %.2f%%\n", best * 100.0f);
    Serial.println("========================================\n");
}
