#include <cmath>
#include <iostream>
#include <vector>

#include <cufft.h>

#include "cuda_check.cuh"
#include "signal_kernels.cuh"

#define CUFFT_CHECK(call)                                                    \
    do {                                                                    \
        cufftResult err = (call);                                          \
        if (err != CUFFT_SUCCESS) {                                        \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__  \
                      << " - code " << err << std::endl;                   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                   \
    } while (0)

namespace {

constexpr int kFftSize = 2048;            // FFT size in samples, rectangular window (no tapering)
constexpr float kSampleRateHz = 8192.0f;   // Fs/N = 4 Hz per bin, so integer-Hz tones land exactly on a bin

constexpr int kNumTones = 4;
constexpr float kToneFreqsHz[kNumTones]    = {200.0f, 440.0f, 880.0f, 1760.0f};
constexpr float kToneAmplitudes[kNumTones] = {1.0f, 0.75f, 0.5f, 0.25f};

}  // namespace

int main() {
    const int numSpectrumBins = kFftSize / 2 + 1;  // R2C output length (Hermitian symmetry)

    float* d_signal = nullptr;
    cufftComplex* d_spectrum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, kFftSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, numSpectrumBins * sizeof(cufftComplex)));

    // Signal generation (the 4-tone mix) runs on the GPU.
    generateSignalOnGpu(d_signal, kFftSize, kSampleRateHz, kToneFreqsHz, kToneAmplitudes, kNumTones);

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, kFftSize, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftExecR2C(plan, d_signal, d_spectrum));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cufftComplex> h_spectrum(numSpectrumBins);
    CUDA_CHECK(cudaMemcpy(h_spectrum.data(), d_spectrum,
                          numSpectrumBins * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // cuFFT's R2C transform is unnormalized: for a tone at an exact bin, the
    // complex magnitude at that bin equals amplitude * kFftSize / 2.
    std::cout << "bin\tfreq(Hz)\tmagnitude" << std::endl;
    for (int bin = 0; bin < numSpectrumBins; ++bin) {
        const float re = h_spectrum[bin].x;
        const float im = h_spectrum[bin].y;
        const float magnitude = std::sqrt(re * re + im * im) / (kFftSize / 2.0f);
        if (magnitude > 0.05f) {
            const float freqHz = bin * kSampleRateHz / kFftSize;
            std::cout << bin << "\t" << freqHz << "\t\t" << magnitude << std::endl;
        }
    }

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_spectrum));

    return 0;
}