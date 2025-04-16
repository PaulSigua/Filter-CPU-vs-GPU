// unified_emboss_filter.cu

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
            exit(code);
    }
}

std::vector<double> createEmbossKernel(int kernel_size)
{
    std::vector<double> kernel(kernel_size * kernel_size, 0.0);
    int half = kernel_size / 2;
    for (int y = 0; y < kernel_size; ++y)
    {
        for (int x = 0; x < kernel_size; ++x)
        {
            if (x < half && y < half)
                kernel[y * kernel_size + x] = -1;
            else if (x > half && y > half || (x == half && y == half))
                kernel[y * kernel_size + x] = 1;
        }
    }
    return kernel;
}

std::vector<double> applyConvolutionCPU(const std::vector<unsigned char> &image, const std::vector<double> &kernel,
                                        int width, int height, int kernel_size)
{
    int half_kernel = kernel_size / 2;
    std::vector<double> result(width * height, 0.0);
    for (int y = half_kernel; y < height - half_kernel; ++y)
    {
        for (int x = half_kernel; x < width - half_kernel; ++x)
        {
            double sum = 0.0;
            for (int ky = -half_kernel; ky <= half_kernel; ++ky)
            {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx)
                {
                    int pixel = image[(y + ky) * width + (x + kx)];
                    sum += pixel * kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                }
            }
            result[y * width + x] = sum;
        }
    }
    return result;
}

__global__ void applyConvolutionGPU(const unsigned char *d_image, const double *d_kernel, double *d_result,
                                    int width, int height, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = kernel_size / 2;

    if (x >= half && x < width - half && y >= half && y < height - half)
    {
        double sum = 0.0;
        for (int ky = -half; ky <= half; ++ky)
        {
            for (int kx = -half; kx <= half; ++kx)
            {
                int img_idx = (y + ky) * width + (x + kx);
                int ker_idx = (ky + half) * kernel_size + (kx + half);
                sum += d_image[img_idx] * d_kernel[ker_idx];
            }
        }
        d_result[y * width + x] = sum;
    }
}

void normalizeAndSave(const std::vector<double> &input, int width, int height, const std::string &filename)
{
    std::vector<unsigned char> output(width * height);
    double min_val = *std::min_element(input.begin(), input.end());
    double max_val = *std::max_element(input.begin(), input.end());

    for (int i = 0; i < width * height; ++i)
    {
        double norm = 255.0 * (input[i] - min_val) / (max_val - min_val);
        output[i] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, norm)));
    }

    stbi_write_jpg(filename.c_str(), width, height, 1, output.data(), 100);
}

void runConvolutionComparison(const std::vector<unsigned char> &gray, int width, int height, int kernel_size)
{
    std::cout << "\n--- Kernel Size: " << kernel_size << "x" << kernel_size << " ---" << std::endl;
    auto kernel = createEmbossKernel(kernel_size);

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto result_cpu = applyConvolutionCPU(gray, kernel, width, height, kernel_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() * 1000.0 << " ms" << std::endl;
    normalizeAndSave(result_cpu, width, height, "resultado_cpu_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size) + ".jpg");

    // GPU
    unsigned char *d_image;
    double *d_kernel;
    double *d_result;
    size_t img_bytes = width * height * sizeof(unsigned char);
    size_t ker_bytes = kernel.size() * sizeof(double);
    size_t res_bytes = width * height * sizeof(double);

    cudaMalloc(&d_image, img_bytes);
    cudaMalloc(&d_kernel, ker_bytes);
    cudaMalloc(&d_result, res_bytes);

    cudaMemcpy(d_image, gray.data(), img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    applyConvolutionGPU<<<grid, block>>>(d_image, d_kernel, d_result, width, height, kernel_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;

    std::vector<double> result_gpu(width * height);
    cudaMemcpy(result_gpu.data(), d_result, res_bytes, cudaMemcpyDeviceToHost);
    normalizeAndSave(result_gpu, width, height, "resultado_gpu_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size) + ".jpg");

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Comparar resultados CPU vs GPU
    int different_pixels = 0;
    double tolerance = 1e-3; // tolerancia para diferencias flotantes
    for (int i = 0; i < width * height; ++i)
    {
        if (std::abs(result_cpu[i] - result_gpu[i]) > tolerance){
            ++different_pixels;
        }
    }
    double error_percentage = 100.0 * different_pixels / (width * height);
    std::cout << std::fixed;
    std::cout.precision(8);
    std::cout << "Error (diferencia de píxeles): " << error_percentage << " %" << std::endl;
}

int main()
{
    int width, height, channels;
    unsigned char *h_image = stbi_load("image2.jpg", &width, &height, &channels, 0);
    if (!h_image || channels < 3)
    {
        std::cerr << "Error al cargar la imagen o formato incorrecto." << std::endl;
        return -1;
    }

    std::vector<unsigned char> gray(width * height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = (y * width + x) * channels;
            gray[y * width + x] = static_cast<unsigned char>(0.299 * h_image[idx] + 0.587 * h_image[idx + 1] + 0.114 * h_image[idx + 2]);
        }
    }

    std::vector<int> kernel_sizes = {9, 13, 21};
    for (int k : kernel_sizes)
    {
        runConvolutionComparison(gray, width, height, k);
    }

    stbi_image_free(h_image);
    return 0;
}
