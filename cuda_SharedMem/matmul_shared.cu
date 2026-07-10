#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Square matrix dimension and tile size (threads per block edge).
#define N 512
#define TILE 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Tiled matrix multiplication C = A * B using shared memory.
//
// Each block computes one TILE x TILE tile of C. The tiles of A and B needed
// for that block are staged into shared memory one TILE-wide slab at a time,
// so every element loaded from global memory is reused TILE times by the
// threads in the block instead of being re-fetched from global memory.
__global__ void matMulShared(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < n / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];

        // Wait until the whole tile is loaded before anyone reads it.
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Wait until everyone is done reading before overwriting the tile.
        __syncthreads();
    }

    C[row * n + col] = acc;
}

static void referenceMatMul(const std::vector<float> &A,
                             const std::vector<float> &B,
                             std::vector<float> &C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) {
                acc += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = acc;
        }
    }
}

int main() {
    static_assert(N % TILE == 0, "N must be a multiple of TILE");

    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> hA(N * N), hB(N * N), hC(N * N), hRef(N * N);
    for (int i = 0; i < N * N; ++i) {
        hA[i] = static_cast<float>(rand() % 10);
        hB[i] = static_cast<float>(rand() % 10);
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matMulShared<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    referenceMatMul(hA, hB, hRef, N);

    double maxAbsErr = 0.0;
    for (int i = 0; i < N * N; ++i) {
        maxAbsErr = std::max(maxAbsErr, static_cast<double>(std::abs(hC[i] - hRef[i])));
    }

    printf("Matrix size: %dx%d, tile size: %dx%d\n", N, N, TILE, TILE);
    printf("Kernel time: %.3f ms\n", ms);
    printf("Max abs error vs. CPU reference: %g\n", maxAbsErr);
    printf(maxAbsErr < 1e-3 ? "Result: PASS\n" : "Result: FAIL\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return maxAbsErr < 1e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
}
