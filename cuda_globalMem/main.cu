#include <iostream>
#include <vector>
#include <cmath>

// Kernel that reads its inputs from and writes its output to global memory.
// Each thread handles one element of the arrays.
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

int main() {
    const int n = 1 << 20; // ~1M elements
    const size_t bytes = n * sizeof(float);

    // Host memory
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate global memory on the device
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy inputs from host to device global memory
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel: each thread reads/writes global memory once
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back from device global memory to host
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            ok = false;
            std::cerr << "Mismatch at index " << i << ": got " << h_c[i]
                      << ", expected " << expected << std::endl;
            break;
        }
    }

    std::cout << (ok ? "Success! " : "Failure! ") << n
              << " elements added using CUDA global memory." << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return ok ? 0 : 1;
}