#include <cmath>

#include "cuda_check.cuh"
#include "signal_kernels.cuh"

namespace {

__global__ void generateSignalKernel(float* signal, int numSamples, float sampleRateHz,
                                      const float* freqsHz, const float* amplitudes,
                                      const float* phasesRad, int numTones) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSamples) return;

    const float t = static_cast<float>(idx) / sampleRateHz;
    float sample = 0.0f;
    for (int tone = 0; tone < numTones; ++tone) {
        // cos() (rather than sin()) so that a tone's FFT bin phase equals its
        // input phase directly, with no constant -90 degree offset to correct for.
        sample += amplitudes[tone] * cosf(2.0f * static_cast<float>(M_PI) * freqsHz[tone] * t + phasesRad[tone]);
    }
    signal[idx] = sample;
}

}  // namespace

void generateSignalOnGpu(float* d_signal, int numSamples, float sampleRateHz,
                          const float* freqsHz, const float* amplitudes,
                          const float* phasesRad, int numTones) {
    float *d_freqs, *d_amps, *d_phases;
    CUDA_CHECK(cudaMalloc(&d_freqs, numTones * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_amps, numTones * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phases, numTones * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_freqs, freqsHz, numTones * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_amps, amplitudes, numTones * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phases, phasesRad, numTones * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    generateSignalKernel<<<blocks, threadsPerBlock>>>(d_signal, numSamples, sampleRateHz,
                                                       d_freqs, d_amps, d_phases, numTones);
    CUDA_CHECK(cudaGetLastError());

    // Kernel launches on the default stream are ordered after these frees,
    // but the frees themselves don't wait for the kernel to finish reading
    // d_freqs/d_amps/d_phases unless we synchronize first.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_freqs));
    CUDA_CHECK(cudaFree(d_amps));
    CUDA_CHECK(cudaFree(d_phases));
}