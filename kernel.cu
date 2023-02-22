#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _CRT_SECURE_NO_WARNINGS
#include "cstdio"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <algorithm>

#define _CRT_SECURE_NO_WARNINGS
#define BLOCK_SIZE 32
#define THREADS_NUM 32
#include <cstdint>


typedef struct
{
    int width;
    int height;
    unsigned char* elements;
} Matrix;

__device__ uint8_t biliner(
    const float tx,
    const float ty,
    const uint8_t c00,
    const uint8_t c10,
    const uint8_t c01,
    const uint8_t c11)
{
    const float color = (1.0f - tx) * (1.0f - ty) * (c00 / 255.0) +
        tx * (1.0f - ty) * (c10 / 255.0) +
        (1.0f - tx) * ty * (c01 / 255.0) +
        tx * ty * (c11 / 255.0);

    return (color * 255);
}

// image resizing kernel
__global__ void ResizeImage(const Matrix A, const Matrix C, const int32_t src_img_width, const int32_t src_img_height, const int32_t dest_img_width, const int32_t dest_img_height)
{
    const int RGB_SIZE = 4;
    const int row = (blockIdx.y * blockDim.y + threadIdx.y);
    const int col = (blockIdx.x * blockDim.x + threadIdx.x);

    if (row < dest_img_height - 1 && col < dest_img_width - 1)
    {
        const float gx = col * (float(src_img_width) / dest_img_width);
        const int gxi = int(gx) * RGB_SIZE;
        const float gy = row * (float(src_img_height) / dest_img_height);
        const int gyi = int(gy);

        const int c00_index = gyi * A.width + gxi;
        const uint8_t c00_1 = A.elements[c00_index]; //R
        const uint8_t c00_2 = A.elements[c00_index + 1]; //G
        const uint8_t c00_3 = A.elements[c00_index + 2]; //B

        const int c10_index = gyi * A.width + (gxi + RGB_SIZE);
        const uint8_t c10_1 = A.elements[c10_index]; //R
        const uint8_t c10_2 = A.elements[c10_index + 1]; //G
        const uint8_t c10_3 = A.elements[c10_index + 2]; //B

        const int c01_index = (gyi + 1) * A.width + gxi;
        const uint8_t c01_1 = A.elements[c01_index]; //R
        const uint8_t c01_2 = A.elements[c01_index + 1]; //G
        const uint8_t c01_3 = A.elements[c01_index + 2]; //B

        const int c11_index = (gyi + 1) * A.width + (gxi + RGB_SIZE);
        const uint8_t c11_1 = A.elements[c11_index]; //R
        const uint8_t c11_2 = A.elements[c11_index + 1]; //G
        const uint8_t c11_3 = A.elements[c11_index + 2]; //B 

        const float tx = gx - int(gx);
        const float ty = gy - gyi;
        const int C_dest = row * C.width + col * RGB_SIZE;

        const uint8_t Cvalue_R = biliner(tx, ty, c00_1, c10_1, c01_1, c11_1);
        C.elements[C_dest] = Cvalue_R;

        const uint8_t Cvalue_G = biliner(tx, ty, c00_2, c10_2, c01_2, c11_2);
        C.elements[C_dest + 1] = Cvalue_G;

        const uint8_t Cvalue_B = biliner(tx, ty, c00_3, c10_3, c01_3, c11_3);
        C.elements[C_dest + 2] = Cvalue_B;

        C.elements[C_dest + 3] = 255; 
    }
    else if (col == dest_img_width - 1 && row < dest_img_height) {
        const int C_dest = (row)*C.width + col * RGB_SIZE;
        
        C.elements[C_dest] = C.elements[C_dest - 3];
        C.elements[C_dest + 1] = C.elements[C_dest - 2];
        C.elements[C_dest + 2] = C.elements[C_dest - 1];
        C.elements[C_dest + 3] = 255;
    }
    else if (col < dest_img_width && row == dest_img_height - 1) {
        const int C_dest = (row)*C.width + col * RGB_SIZE;
        C.elements[C_dest] = C.elements[C_dest - C.width];
        C.elements[C_dest + 1] = C.elements[C_dest + 1 - C.width];
        C.elements[C_dest + 2] = C.elements[C_dest + 2 - C.width];
        C.elements[C_dest + 3] = 255;
    }
}

// gaussian blur kernel
__global__ void gaussianBlurKernel(Matrix input, Matrix output, int width, int height, float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * 4;

    float4 sum = make_float4(0, 0, 0, 0);
    float weightSum = 0;

    int kernelRadius = (int)ceil(3.0f * sigma);

    // realising gaussian blur algorithm
    if ((x + width * y) < width * height && (x + height * y) < width * height) {
        for (int i = -kernelRadius; i <= kernelRadius; i++)
        {
            for (int j = -kernelRadius; j <= kernelRadius; j++)
            {
                int ii = min(max(y + i, 0), height - 1);
                int jj = min(max(x + j, 0), width - 1);

                int neighbourIndex = (ii * width + jj) * 4;
                unsigned char r = input.elements[neighbourIndex];
                unsigned char g = input.elements[neighbourIndex + 1];
                unsigned char b = input.elements[neighbourIndex + 2];
                unsigned char a = input.elements[neighbourIndex + 3];

                float weight = exp(-((i * i + j * j) / (2 * sigma * sigma)));
                sum.x += weight * r;
                sum.y += weight * g;
                sum.z += weight * b;
                sum.w += weight * a;
                weightSum += weight;
            }
        }

        output.elements[idx] = (unsigned char)(sum.x / weightSum);
        output.elements[idx + 1] = (unsigned char)(sum.y / weightSum);
        output.elements[idx + 2] = (unsigned char)(sum.z / weightSum);
        output.elements[idx + 3] = (unsigned char)(sum.w / weightSum);
    }
}

// median blur kernel
__global__ void medianBlurKernel(Matrix input, Matrix output, int width, int height)
{
    const int kernelSize = 7;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * 4;

    int kernelRadius = kernelSize / 2;

    unsigned char neighboursR[kernelSize * kernelSize];
    unsigned char neighboursG[kernelSize * kernelSize];
    unsigned char neighboursB[kernelSize * kernelSize];
    unsigned char neighboursA[kernelSize * kernelSize];

    int neighbourCount = 0;
    if ((x + width * y) < width * height && (x + height * y) < width * height) {
        for (int i = -kernelRadius; i <= kernelRadius; i++)
        {
            for (int j = -kernelRadius; j <= kernelRadius; j++)
            {
                int ii = min(max(y + i, 0), height - 1);
                int jj = min(max(x + j, 0), width - 1);

                int neighbourIndex = (ii * width + jj) * 4;
                neighboursR[neighbourCount] = input.elements[neighbourIndex];
                neighboursG[neighbourCount] = input.elements[neighbourIndex + 1];
                neighboursB[neighbourCount] = input.elements[neighbourIndex + 2];
                neighboursA[neighbourCount] = input.elements[neighbourIndex + 3];
                neighbourCount++;
            }
        }

        for (int i = 0; i < (kernelSize * kernelSize); i++) {
            for (int j = i + 1; j < (kernelSize * kernelSize); j++) {
                if (neighboursR[i] > neighboursR[j]) {
                    char tmpR = neighboursR[i];
                    neighboursR[i] = neighboursR[j];
                    neighboursR[j] = tmpR;
                }
                if (neighboursG[i] > neighboursG[j]) {
                    char tmpG = neighboursG[i];
                    neighboursG[i] = neighboursG[j];
                    neighboursG[j] = tmpG;
                }
                if (neighboursB[i] > neighboursB[j]) {
                    char tmpB = neighboursB[i];
                    neighboursB[i] = neighboursB[j];
                    neighboursB[j] = tmpB;
                }
                if (neighboursA[i] > neighboursA[j]) {
                    char tmpA = neighboursA[i];
                    neighboursA[i] = neighboursA[j];
                    neighboursA[j] = tmpA;
                }
            }
        }

        int medianIndex = neighbourCount / 2;
        output.elements[idx] = neighboursR[medianIndex];
        output.elements[idx + 1] = neighboursG[medianIndex];
        output.elements[idx + 2] = neighboursB[medianIndex];
        output.elements[idx + 3] = neighboursA[medianIndex];
    }

}

// sobel filter kernel
__global__ void sobelFilterKernel(Matrix input, Matrix output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * 4;

    int gx = input.elements[idx - width * 4 - 4] + 2 * input.elements[idx - 4] + input.elements[idx + width * 4 - 4]
        - input.elements[idx - width * 4 + 4] - 2 * input.elements[idx + 4] - input.elements[idx + width * 4 + 4];
    int gy = input.elements[idx - width * 4 - 4] + 2 * input.elements[idx - width * 4] + input.elements[idx - width * 4 + 4]
        - input.elements[idx + width * 4 - 4] - 2 * input.elements[idx + width * 4] - input.elements[idx + width * 4 + 4];

    int g = sqrtf(gx * gx + gy * gy);
    output.elements[idx] = g;
    output.elements[idx + 1] = g;
    output.elements[idx + 2] = g;
    output.elements[idx + 3] = 255;
}

extern "C" unsigned char* kernel_wrapper(unsigned char* data, unsigned char* median, unsigned char* gaussian, unsigned char* sobel, unsigned char* resized, int d_w, int d_h)
{
    int32_t rgb_size = 4; 

    // size of original image
    int32_t src_img_width = 880;
    int32_t src_img_height = 880;
    uint64_t src_size = src_img_height * src_img_width * rgb_size * sizeof(uint8_t);

    // size of resized image
    int32_t dest_img_width = d_w;
    int32_t dest_img_height = d_h;
    uint64_t dest_size = dest_img_height * dest_img_width * rgb_size * sizeof(uint8_t);

    // CPU declarations
    // in image
    Matrix in_cpu_image;
    in_cpu_image.width = src_img_width * rgb_size;
    in_cpu_image.height = src_img_height;
    in_cpu_image.elements = data;

    //resized image
    Matrix resized_cpu_image;
    resized_cpu_image.width = dest_img_width * rgb_size;
    resized_cpu_image.height = dest_img_height;
    resized_cpu_image.elements = (uint8_t*)malloc(dest_size);

    // median blue image
    Matrix median_cpu_image;
    median_cpu_image.width = src_img_width * rgb_size;
    median_cpu_image.height = src_img_height;
    median_cpu_image.elements = (uint8_t*)malloc(src_size);

    // gaussian blur image
    Matrix gaussian_cpu_image;
    gaussian_cpu_image.width = src_img_width * rgb_size;
    gaussian_cpu_image.height = src_img_height;
    gaussian_cpu_image.elements = (uint8_t*)malloc(src_size);

    // sobel filter image
    Matrix sobel_cpu_image;
    sobel_cpu_image.width = src_img_width * rgb_size;
    sobel_cpu_image.height = src_img_height;
    sobel_cpu_image.elements = (uint8_t*)malloc(src_size);

    // GPU declarations 
    // in image
    Matrix in_gpu_image;
    in_gpu_image.width = in_cpu_image.width;
    in_gpu_image.height = in_cpu_image.height;
    cudaMalloc(&in_gpu_image.elements, src_size);
    cudaMemcpy(in_gpu_image.elements, in_cpu_image.elements, src_size, cudaMemcpyHostToDevice);

    // resized image
    Matrix resized_gpu_image;
    resized_gpu_image.width = resized_cpu_image.width;
    resized_gpu_image.height = resized_cpu_image.height;
    cudaMalloc(&resized_gpu_image.elements, dest_size);

    // median blur image
    Matrix median_gpu_image;
    median_gpu_image.width = in_cpu_image.width;
    median_gpu_image.height = in_cpu_image.height;
    cudaMalloc(&median_gpu_image.elements, src_size);

    // gaussian blur image
    Matrix gaussian_gpu_image;
    gaussian_gpu_image.width = in_cpu_image.width;
    gaussian_gpu_image.height = in_cpu_image.height;
    cudaMalloc(&gaussian_gpu_image.elements, src_size);

    // sobel filter image
    Matrix sobel_gpu_image;
    sobel_gpu_image.width = in_cpu_image.width;
    sobel_gpu_image.height = in_cpu_image.height;
    cudaMalloc(&sobel_gpu_image.elements, src_size);

    // standart deviation to gaussian blur
    float sigma = 2.5;

    // grid to median and gauss
    int xDim = (src_img_width % THREADS_NUM == 0) ? (int)(src_img_width / THREADS_NUM) : (int)ceil(src_img_width / THREADS_NUM) + 1;
    int yDim = (src_img_height % THREADS_NUM == 0) ? (int)(src_img_height / THREADS_NUM) : (int)ceil(src_img_height / THREADS_NUM) + 1;

    dim3 threadsPerBlock(xDim, yDim);
    dim3 numBlocks(THREADS_NUM, THREADS_NUM);

    // grid to resize
    dim3 threadsPerBlockRes(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocksRes(dest_img_width / threadsPerBlock.x, dest_img_height / threadsPerBlock.y);

    printf("GPU operations started\n");

    // kernels
    ResizeImage << <numBlocksRes, threadsPerBlockRes >> > (in_gpu_image, resized_gpu_image, src_img_width, src_img_height, dest_img_width, dest_img_height);

    gaussianBlurKernel << <numBlocks, threadsPerBlock >> > (in_gpu_image, gaussian_gpu_image, src_img_width, src_img_height, sigma);

    medianBlurKernel << <numBlocks, threadsPerBlock >> > (in_gpu_image, median_gpu_image, src_img_width, src_img_height);

    sobelFilterKernel << <numBlocks, threadsPerBlock >> > (in_gpu_image, sobel_gpu_image, src_img_width, src_img_height);
    
    // copy results to host
    cudaMemcpy(median_cpu_image.elements, median_gpu_image.elements, src_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gaussian_cpu_image.elements, gaussian_gpu_image.elements, src_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(sobel_cpu_image.elements, sobel_gpu_image.elements, src_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(resized_cpu_image.elements, resized_gpu_image.elements, dest_size, cudaMemcpyDeviceToHost);

    median = median_cpu_image.elements;
    resized = resized_cpu_image.elements;
    gaussian = gaussian_cpu_image.elements;
    sobel = sobel_cpu_image.elements;

    // write images to files
    stbi_write_png("resized.png", dest_img_width, dest_img_height, 4, resized, dest_img_width * 4);
    printf("resized.png saved\n");
    stbi_write_png("gaussian.png", src_img_width, src_img_height, 4, gaussian, src_img_width * 4);
    printf("gaussian.png saved\n");
    stbi_write_png("sobel.png", src_img_width, src_img_height, 4, sobel, src_img_width * 4);
    printf("sobel.png saved\n");

    // cuda free
    cudaFree(in_gpu_image.elements);
    cudaFree(resized_gpu_image.elements);
    cudaFree(median_gpu_image.elements);
    cudaFree(gaussian_gpu_image.elements);
    cudaFree(sobel_gpu_image.elements);

    // free
    free(in_cpu_image.elements);
    free(resized_cpu_image.elements);
    free(gaussian_cpu_image.elements);
    free(sobel_cpu_image.elements);

    return median;
}

int main()
{
    // input image size
    int s_w = 880;
    int s_h = 880;

    // resize scale
    float res = 2.0;
    int d_w = int(res * 880.);
    int d_h = int(res * 880.);

    int channels;
    unsigned char* img = stbi_load("neon.png", &s_w, &s_h, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        return 0;
    }
    printf("Image loaded\n", s_w, s_h, channels);

    unsigned char* resized = (unsigned char*)malloc((d_w * d_h * 4) * sizeof(unsigned char));
    unsigned char* median = (unsigned char*)malloc((s_w * s_h * 4) * sizeof(unsigned char));
    unsigned char* gaussian = (unsigned char*)malloc((s_w * s_h * 4) * sizeof(unsigned char));
    unsigned char* sobel = (unsigned char*)malloc((s_w * s_h * 4) * sizeof(unsigned char));
    median = kernel_wrapper(img, median, gaussian, sobel, resized, d_w, d_h);

    stbi_write_png("median.png", s_w, s_h, channels, median, s_w * channels);
    printf("madian.png saved\n");

    free(median);
    free(gaussian);
    free(resized);
    free(sobel);

    return 0;
}


