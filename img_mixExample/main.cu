#include <iostream>
#include <cstdio>
#include <string>
#include <jpeglib.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;

int readJPEG(const std::string filename, uchar*& r, uchar*& g, uchar*& b, int& width, int& height) {
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        return -1;
    }

    jpeg_decompress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return -1;
    }

    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    if (cinfo.output_components != 3) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return -1;
    }

    width = static_cast<int>(cinfo.output_width);
    height = static_cast<int>(cinfo.output_height);

    r = new uchar[width * height];
    g = new uchar[width * height];
    b = new uchar[width * height];

    const int rowStride = width * cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)(
        reinterpret_cast<j_common_ptr>(&cinfo), JPOOL_IMAGE, rowStride, 1);

    while (cinfo.output_scanline < cinfo.output_height) {
        const int row = static_cast<int>(cinfo.output_scanline);
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int x = 0; x < width; ++x) {
            const int pixel = row * width + x;
            r[pixel] = buffer[0][x * 3 + 0];
            g[pixel] = buffer[0][x * 3 + 1];
            b[pixel] = buffer[0][x * 3 + 2];
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);

    return 0;
}

int saveJPEG(const std::string filename, uchar* r, uchar* g, uchar* b, const int width, const int height, const int quality = 90) {
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        return -1;
    }

    jpeg_compress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    const int rowStride = width * cinfo.input_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)(
        reinterpret_cast<j_common_ptr>(&cinfo), JPOOL_IMAGE, rowStride, 1);

    while (cinfo.next_scanline < cinfo.image_height) {
        const int row = static_cast<int>(cinfo.next_scanline);
        for (int x = 0; x < width; ++x) {
            const int pixel = row * width + x;
            buffer[0][x * 3 + 0] = r[pixel];
            buffer[0][x * 3 + 1] = g[pixel];
            buffer[0][x * 3 + 2] = b[pixel];
        }
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(file);

    return 0;
}

__constant__ float d_p1;
__constant__ float d_p2;

__global__ void toGrayScaleKernel(uchar* r, const uchar* g, const uchar* b, int width, int height) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        r[idx] = static_cast<uchar>(0.299f * static_cast<float>(r[idx]) + 0.587f * static_cast<float>(g[idx]) + 0.114f * static_cast<float>(b[idx]));
    }
}

__global__ void mixKernel(uchar* c1, uchar* c2, int width, int height) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        c1[idx] = static_cast<uchar>(d_p1 * static_cast<float>(c1[idx]) + d_p2 * static_cast<float>(c2[idx]));
    }
}

void gpu_toGrayScale(uchar* r, uchar* g, uchar* b, int width, int height) {
    const size_t numBytes = static_cast<size_t>(width) * height * sizeof(uchar);

    uchar *d_r, *d_g, *d_b;
    cudaMalloc(&d_r, numBytes);
    cudaMalloc(&d_g, numBytes);
    cudaMalloc(&d_b, numBytes);

    cudaMemcpy(d_r, r, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, g, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, numBytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (width * height + blockSize - 1) / blockSize;
    toGrayScaleKernel<<<gridSize, blockSize>>>(d_r, d_g, d_b, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(r, d_r, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
}

void gpu_mix(uchar* c1, uchar* c2, int width, int height, float p1, float p2)
{
    const size_t numBytes = static_cast<size_t>(width) * height * sizeof(uchar);

    uchar *d_c1, *d_c2;
    cudaMalloc(&d_c1, numBytes);
    cudaMalloc(&d_c2, numBytes);

    cudaMemcpy(d_c1, c1, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2, c2, numBytes, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_p1, &p1, sizeof(float));
    cudaMemcpyToSymbol(d_p2, &p2, sizeof(float));

    const int blockSize = 256;
    const int gridSize = (width * height + blockSize - 1) / blockSize;
    mixKernel<<<gridSize, blockSize>>>(d_c1, d_c2, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(c1, d_c1, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_c1);
    cudaFree(d_c2);
}

void cpu_mix(uchar* r1, uchar* g1, uchar* b1, uchar* r2, uchar* g2, uchar* b2, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sr1 = static_cast<float>(r1[x + y * width]) + static_cast<float>(g1[x + y * width]) + static_cast<float>(b1[x + y * width]);
            float sr2 = static_cast<float>(r2[x + y * width]) + static_cast<float>(g2[x + y * width]) + static_cast<float>(b2[x + y * width]);
            r1[x + y * width] = static_cast<uchar>(sr1/6.0) + static_cast<uchar>(sr2/6.0);
            g1[x + y * width] = static_cast<uchar>(sr1/6.0) + static_cast<uchar>(sr2/6.0);
            b1[x + y * width] = static_cast<uchar>(sr1/6.0) + static_cast<uchar>(sr2/6.0);
        }
    }
}

int main(int nArgs, char* chArgs[]) {
    //std::cout << "Hello, World!" << std::endl;

    float p1 = 0.5f;
    float p2 = 0.5f;

    if (nArgs<4) {
        std::cout << "Need at least names of an inputs and output image\n";
        return -1;
    }
    if (nArgs>=4) {
        p1 = std::stof(chArgs[4]);
        std::cout << "p1: " << p1 << "\n";
    }

    if (nArgs>=5) {
        p2 = std::stof(chArgs[5]);
        std::cout << "p2: " << p2 << "\n";
    }

    int width1 = -1;
    int height1 = -1;
    uchar* r1 = NULL;
    uchar* g1 = NULL;
    uchar* b1 = NULL;

    int width2 = -1;
    int height2 = -1;
    uchar* r2 = NULL;
    uchar* g2 = NULL;
    uchar* b2 = NULL;

    if (readJPEG(chArgs[1], r1, g1, b1, width1, height1)==-1) {
        std::cout << "readJPEG error\n";
        return -1;
    }

    std::cout << "readJPEG1 success\n";
    std::cout << "width1: " << width1 << "\n";
    std::cout << "height1: " << height1 << "\n";

    if (readJPEG(chArgs[2], r2, g2, b2, width2, height2)==-1) {
        std::cout << "readJPEG error\n";
        return -1;
    }

    std::cout << "readJPEG2 success\n";
    std::cout << "width2: " << width2 << "\n";
    std::cout << "height2: " << height2 << "\n";

    // image size must be equal
    if ((width1==width2) && (height1==height2)) {

        gpu_toGrayScale(r1, g1, b1, width1, height1);
        gpu_toGrayScale(r2, g2, b2, width2, height2);

        gpu_mix(r1, r2, width1, height1, p1, p2);

        for (int y=0; y < height1; ++y) {
            for (int x=0; x < width1; ++x) {
                //r1[x + y * width1] = static_cast<uchar>((static_cast<float>(r1[x + y * width1]) + static_cast<float>(r2[x + y * width1])) / 2.0f);
                g1[x + y * width1] = r1[x + y * width1];
                b1[x + y * width1] = r1[x + y * width1];
            }
        }

        // gray scale data is in r1 and r2 respectively

        //cpu_mix(r1, g1, b1, r2, g2, b2, width1, height1);
        saveJPEG(chArgs[3], r1, g1, b1, width1, height1);
    }
    else {
        std::cout << "Image size must match\n";
    }

    delete[] r1;
    delete[] g1;
    delete[] b1;
    delete[] r2;
    delete[] g2;
    delete[] b2;

    return 0;
}