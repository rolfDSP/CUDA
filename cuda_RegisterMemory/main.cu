#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            std::exit(EXIT_FAILURE);                                    \
        }                                                                \
    } while (0)

// Each thread walks a strided chunk of the input and keeps a running
// total in "sum". Because sum/v are plain scalar locals (no indexing,
// no address taken), the compiler keeps them in registers for the
// thread's whole lifetime instead of spilling to local/global memory.
// Global memory is touched once per input element on read, and once
// per thread on write - the accumulation itself costs no memory traffic.
__global__ void sumOfSquaresKernel(const float* in, float* partialSums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float v = in[i];
        sum += v * v;
    }

    partialSums[idx] = sum;
}

int main() {
    const int n = 1 << 20;
    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i % 100) * 0.01f;

    const int threads = 256;
    const int blocks = 128;
    const int numThreadsTotal = threads * blocks;

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, numThreadsTotal * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    sumOfSquaresKernel<<<blocks, threads>>>(d_in, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partial(numThreadsTotal);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, numThreadsTotal * sizeof(float), cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (float p : h_partial) total += p;

    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = (i % 100) * 0.01;
        expected += v * v;
    }

    std::cout << "GPU result:      " << total << std::endl;
    std::cout << "Expected result: " << expected << std::endl;
    std::cout << "Difference:      " << std::fabs(total - expected) << std::endl;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
