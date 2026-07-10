#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

// Constant memory: small, read-only data cached and broadcast to all
// threads in a warp that read the same address in the same cycle.
// Ideal for filter coefficients, lookup tables, etc.
#define FILTER_WIDTH 5
__constant__ float d_filter[FILTER_WIDTH];

// 1D convolution: each thread computes one output element by combining
// FILTER_WIDTH neighboring input elements, weighted by d_filter.
// Every thread reads the same d_filter values, so the constant cache
// serves them far more efficiently than global memory would.
__global__ void convolve1D(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int radius = FILTER_WIDTH / 2;
    float sum = 0.0f;
    for (int k = 0; k < FILTER_WIDTH; ++k) {
        int idx = i + k - radius;
        if (idx >= 0 && idx < n) {
            sum += input[idx] * d_filter[k];
        }
    }
    output[i] = sum;
}

int main() {
    const int n = 16;
    std::vector<float> h_input(n);
    for (int i = 0; i < n; ++i) h_input[i] = static_cast<float>(i);

    // Simple 5-tap moving-average filter.
    float h_filter[FILTER_WIDTH] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

    // Upload the filter once into constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, sizeof(h_filter)));

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float),
                           cudaMemcpyHostToDevice));

    int threadsPerBlock = 8;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    convolve1D<<<blocks, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(n);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float),
                           cudaMemcpyDeviceToHost));

    std::cout << "Input:  ";
    for (float v : h_input) std::cout << v << " ";
    std::cout << "\nOutput: ";
    for (float v : h_output) std::cout << v << " ";
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}