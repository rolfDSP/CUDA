#include <iostream>
#include <cmath>
#include <chrono>

// Plain serial CPU reference implementation, used as the timing baseline.
void vectorAddCpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Kernel that reads its inputs from and writes its output to global memory.
// Each thread handles one element of the arrays. Identical in spirit to a
// plain vector-add kernel - the interesting part for streaming is how the
// launches below are issued and overlapped with transfers.
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
    const int n = 1 << 22;              // ~4M elements
    const size_t bytes = n * sizeof(float);

    const int numStreams = 4;
    const int chunkSize = n / numStreams;
    const size_t chunkBytes = chunkSize * sizeof(float);

    // Pinned (page-locked) host memory is required for cudaMemcpyAsync to
    // actually run asynchronously with respect to the host and to overlap
    // with kernel execution on the device.
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));
    auto* h_c_cpu = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // --- CPU baseline ---
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorAddCpu(h_a, h_b, h_c_cpu, n);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaStream_t streams[numStreams];
    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    const int threadsPerBlock = 256;
    const int blocksPerChunk = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    // cudaEvent_t timing measures actual device-side elapsed time, unlike a
    // host-side chrono timer around async calls (which would only measure
    // how fast the driver could enqueue work, not how long the GPU took).
    cudaEvent_t gpuStart, gpuStop;
    CUDA_CHECK(cudaEventCreate(&gpuStart));
    CUDA_CHECK(cudaEventCreate(&gpuStop));
    CUDA_CHECK(cudaEventRecord(gpuStart));

    // For each stream: copy its chunk H2D, launch the kernel on that chunk,
    // then copy its chunk D2H - all issued asynchronously. Because each
    // stage lives in its own stream, the GPU can overlap the D2H copy of
    // chunk k-1 with the kernel of chunk k, and that kernel with the H2D
    // copy of chunk k+1 (hardware/driver permitting).
    for (int s = 0; s < numStreams; ++s) {
        int offset = s * chunkSize;

        CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset, chunkBytes,
                                    cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset, chunkBytes,
                                    cudaMemcpyHostToDevice, streams[s]));

        vectorAdd<<<blocksPerChunk, threadsPerBlock, 0, streams[s]>>>(
            d_a + offset, d_b + offset, d_c + offset, chunkSize);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset, chunkBytes,
                                    cudaMemcpyDeviceToHost, streams[s]));
    }

    // Wait for every stream to finish all its queued work.
    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

    CUDA_CHECK(cudaEventRecord(gpuStop));
    CUDA_CHECK(cudaEventSynchronize(gpuStop));
    float gpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c[i] - h_c_cpu[i]) > 1e-5f) {
            ok = false;
            std::cerr << "Mismatch at index " << i << ": GPU got " << h_c[i]
                      << ", CPU expected " << h_c_cpu[i] << std::endl;
            break;
        }
    }

    std::cout << (ok ? "Success! " : "Failure! ") << n
              << " elements added using " << numStreams
              << " CUDA streams." << std::endl;
    std::cout << "CPU time: " << cpuMs << " ms" << std::endl;
    std::cout << "GPU time: " << gpuMs << " ms (includes H2D/D2H transfers)" << std::endl;
    std::cout << "Speedup:  " << (cpuMs / gpuMs) << "x" << std::endl;

    CUDA_CHECK(cudaEventDestroy(gpuStart));
    CUDA_CHECK(cudaEventDestroy(gpuStop));

    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    delete[] h_c_cpu;

    return ok ? 0 : 1;
}
