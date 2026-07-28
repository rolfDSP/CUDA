# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This project should generate a 4-tone sinusoidal mixed signal with different frequencies and amplitudes for each 
tone component

Using the cuFFT library from NVIDIA it should perform a time to frequency domain transform on the GPU.
FFT size is 2048 samples and a rectangular window.

The numerical format of the input signal and the frequency domain output is float.
Input is real, while the FFT output is complex.

## Project state

Implemented. Generates a 4-tone sinusoidal mix on the GPU and runs a cuFFT R2C transform on it.

## Build

```
cmake -B cmake-build-debug -S .
cmake --build cmake-build-debug
```

The build produces a `simpleSigFreq` executable in the `exe/` subdirectory. There is no test suite or lint configuration.

## Architecture

- `CMakeLists.txt` — CMake config (CMake 4.1+, C++20/CUDA 20), builds `simpleSigFreq` from `main.cpp` + `main.cu`, links `CUDA::cufft`, outputs to `exe/`.
- `main.cu` — `generateSignalOnGpu()`: copies tone freqs/amplitudes to the device and launches `generateSignalKernel`, which sums the 4 sinusoids per-sample on the GPU (rectangular window, i.e. no tapering).
- `main.cpp` — entry point: allocates the real input buffer (2048 floats) and complex output buffer (`kFftSize/2+1` `cufftComplex`, R2C Hermitian symmetry), calls `generateSignalOnGpu`, runs `cufftPlan1d`/`cufftExecR2C`, then prints magnitude/frequency for bins above a threshold.
- `signal_kernels.cuh` — shared declaration of `generateSignalOnGpu`, included by both `main.cpp` (caller) and `main.cu` (definition, needs nvcc for the kernel launch).
- `cuda_check.cuh` — shared `CUDA_CHECK` error-checking macro; `main.cpp` additionally defines a local `CUFFT_CHECK` macro for cuFFT return codes.

Tone parameters (`kToneFreqsHz`/`kToneAmplitudes` in `main.cpp`) are chosen as multiples of the bin resolution (`kSampleRateHz / kFftSize` = 4 Hz) so each tone lands exactly on an FFT bin with no spectral leakage.
