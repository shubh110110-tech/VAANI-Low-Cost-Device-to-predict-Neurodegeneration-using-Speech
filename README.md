VAANI is an edge AI-powered voice diagnostic device that detects neurological conditions (ALS and Parkinson's Disease) from speech samples in real-time on a microcontroller.
The Hardware Stack

ESP32 microcontroller (the brains)
INMP441 MEMS microphone (captures voice)
1.3" OLED display (shows results)
SD card module (data logging)
3D-printed enclosure (the physical box)

All running on battery, locally — no cloud dependency.
The ML Pipeline

Input: Raw audio (speech samples)
Feature extraction: MFCC (Mel-Frequency Cepstral Coefficients) — 13 coefficients per frame, designed specifically for human speech
Model: 1D CNN trained on 1,100+ samples from Synapse VOC-ALS dataset + KaggleHub
Output: Classification into 3 classes: ALS, Healthy Voice, Parkinson's Disease
Quantization: INT8 (8-bit integer) so it fits on the ESP32 with only 15KB RAM usage

Performance

Best float32 model: 92.1% accuracy
INT8 quantized (actual device): 88.7% accuracy
Current deployed version: 86% real-world accuracy
Processing time: 144ms per sample (fast enough for live feedback)

Non-invasive, portable neurological screening tool for early detection. Could eventually be deployed in clinics, home monitoring, or as a screening tool in low-resource settings.
