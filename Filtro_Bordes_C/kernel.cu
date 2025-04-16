#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>   
#include <sstream>
#include <algorithm>
#include <omp.h>

#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)       \
                      << " en " << __FILE__ << ":" << __LINE__ << "\n";  \
            exit(1);                                                      \
        }                                                                 \
    }

float* generar_kernel_bordes(int size) {
    float* kernel = new float[size * size];
    int center = size / 2;
    int total = size * size;

    for (int i = 0; i < total; ++i) {
        kernel[i] = 1.0f; 
    }

    kernel[center * size + center] = -1.0f * (total - 1);

    return kernel;
}

inline int clamp(int val, int low, int high) {
    return val < low ? low : (val > high ? high : val);
}

void convolucion_cpu(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernel_size) {

    int k_half = kernel_size / 2;

    
#pragma omp parallel for collapse(3)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;

                
#pragma omp simd collapse(2) reduction(+:sum)
                for (int ky = -k_half; ky <= k_half; ++ky) {
                    for (int kx = -k_half; kx <= k_half; ++kx) {
                        int px = clamp(x + kx, 0, width - 1);
                        int py = clamp(y + ky, 0, height - 1);

                        int img_idx = (py * width + px) * channels + c;
                        int k_idx = (ky + k_half) * kernel_size + (kx + k_half);

                        sum += input[img_idx] * kernel[k_idx];
                    }
                }

                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = static_cast<unsigned char>(clamp(static_cast<int>(sum), 0, 255));
            }
        }
    }
}

__global__ void convolucion_gpu(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernel_size) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k_half = kernel_size / 2;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;

        for (int ky = -k_half; ky <= k_half; ++ky) {
            for (int kx = -k_half; kx <= k_half; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                int img_idx = (py * width + px) * channels + c;
                int k_idx = (ky + k_half) * kernel_size + (kx + k_half);

                sum += input[img_idx] * kernel[k_idx];
            }
        }

        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = static_cast<unsigned char>(min(max(int(sum), 0), 255));
    }
}

void ejecutar_gpu(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernel_size) {

    unsigned char* d_input;
    unsigned char* d_output;
    float* d_kernel;

    size_t input_size = width * height * channels * sizeof(unsigned char);
    size_t kernel_size_in_bytes = kernel_size * kernel_size * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, input_size));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_size_in_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel, kernel_size_in_bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();

    convolucion_gpu << <gridDim, blockDim >> > (d_input, d_output, width, height, channels, d_kernel, kernel_size);

    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "GPU tiempo (" << kernel_size << "x" << kernel_size << "): " << elapsed.count() << " ms\n";

    CHECK_CUDA(cudaMemcpy(output, d_output, input_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

void ejecutar_cpu(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernel_size) {

    auto start = std::chrono::high_resolution_clock::now();
    convolucion_cpu(input, output, width, height, channels, kernel, kernel_size);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CPU tiempo (" << kernel_size << "x" << kernel_size << "): "
        << elapsed.count() << " ms\n";
}

int contar_diferencias_pixeles(const unsigned char* img1, const unsigned char* img2, int width, int height, int channels) {
    int diferencias = 0;

    for (int i = 0; i < width * height; ++i) {
        bool distinto = false;
        for (int c = 0; c < channels; ++c) {
            if (img1[i * channels + c] != img2[i * channels + c]) {
                distinto = true;
                break;
            }
        }
        if (distinto) {
            ++diferencias;
        }
    }

    return diferencias;
}


int main() {
    const char* filename = "image2.jpg";

    int width, height, channels;
    unsigned char* h_img = stbi_load(filename, &width, &height, &channels, 0);
    if (!h_img) {
        std::cerr << "No se pudo leer la imagen.\n";
        return 1;
    }

    std::cout << "Imagen cargada: " << width << "x" << height
        << ", Canales: " << channels << "\n";

    size_t size = width * height * channels;
    int kernel_sizes[] = {9, 13, 21};
    int num_kernels = sizeof(kernel_sizes) / sizeof(kernel_sizes[0]);

    for (int i = 0; i < num_kernels; ++i) {

        int k_size = kernel_sizes[i];
        float* kernel = generar_kernel_bordes(k_size);

        unsigned char* cpu_output = new unsigned char[size];
        unsigned char* gpu_output = new unsigned char[size];

        //Procesar con CPU
        std::cout << "\nProcesando con kernel " << k_size << "x" << k_size << "\n";
        ejecutar_cpu(h_img, cpu_output, width, height, channels, kernel, k_size);
        std::string cpu_name = "cpu_" + std::to_string(k_size) + "x" + std::to_string(k_size) + ".jpg";
        if (!stbi_write_jpg(cpu_name.c_str(), width, height, channels, cpu_output, 90)) {
            std::cerr << "Error al guardar alguna imagen.\n";
        }
        else {
            std::cout << "Imagen guardada como: " << cpu_name << "\n";
        }

        // Procesar con GPU
        ejecutar_gpu(h_img, gpu_output, width, height, channels, kernel, k_size);
        std::string gpu_name = "gpu_" + std::to_string(k_size) + "x" + std::to_string(k_size) + ".jpg";
        if (!stbi_write_jpg(gpu_name.c_str(), width, height, channels, gpu_output, 90)) {
            std::cerr << "Error al guardar la imagen de GPU.\n";
        }
        else {
            std::cout << "Imagen guardada (GPU): " << gpu_name << "\n";
        }

        // Comparar resultados CPU vs GPU (pixel a pixel)
        int errores_pixel = contar_diferencias_pixeles(cpu_output, gpu_output, width, height, channels);
        float porcentaje_error = 100.0f * errores_pixel / (width * height);

        std::cout << "Diferencias por píxel con kernel " << k_size << "x" << k_size
            << ": " << errores_pixel << " píxeles (" << porcentaje_error << "% del total)\n";

        delete[] cpu_output;
        delete[] gpu_output;
        delete[] kernel;
    }

    stbi_image_free(h_img);
    return 0;
}
