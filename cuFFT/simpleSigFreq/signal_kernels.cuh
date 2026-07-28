#pragma once

// Launches a CUDA kernel that fills d_signal[0..numSamples) with the sum of
// numTones sinusoidal tones sampled at sampleRateHz. Runs entirely on the GPU.
// freqsHz, amplitudes and phasesRad are host arrays of length numTones.
void generateSignalOnGpu(float* d_signal, int numSamples, float sampleRateHz,
                          const float* freqsHz, const float* amplitudes,
                          const float* phasesRad, int numTones);