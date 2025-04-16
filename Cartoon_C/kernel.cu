// main.cu
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>              // Para std::fixed y std::setprecision
#include <cuda_runtime.h>
#include <thread>
#include <vector>

// Incluir las librerías de stb para carga/escritura de imágenes
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3  // Usamos escala de colores

// Definición de PI para Visual Studio Community
#ifndef PI
#define PI 3.14159265358979323846
#endif

// Definición de macros para minimizar sobrecarga en funciones de min/max
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

// ---------------------------------------------------------------------
// Generar filtro Laplacian of Gaussian (LoG) de tamaño NxN con sigma dado
// ---------------------------------------------------------------------
void generateLoGFilter(float* mask, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;

    // Se usa la fórmula del LoG 2D:
    // LoG(x,y) = -1/(pi*sigma^4) * [1 - (x^2+y^2)/(2sigma^2)] * exp(-(x^2+y^2)/(2sigma^2))
    for (int j = -half; j <= half; ++j) {
        for (int i = -half; i <= half; ++i) {
            float r2 = static_cast<float>(i * i + j * j);
            float factor = -1.0f / (PI * std::pow(sigma, 4));
            float expo = expf(-r2 / (2.0f * sigma * sigma));
            float value = factor * (1.0f - (r2 / (2.0f * sigma * sigma))) * expo;
            mask[(j + half) * size + (i + half)] = value;
            sum += value;
        }
    }

}

// ---------------------------------------------------------------------
// CPU - Función que aplica el filtro Cartoon + Laplaciano unificado de manera paralela
// ---------------------------------------------------------------------
void cartoonLaplaceCPU(const unsigned char* input,
    unsigned char* output,
    float* lapMask,
    int width, int height,
    int maskSize,
    int quantStep,
    float threshold)
{
    int half = maskSize / 2;
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // fallback por si hardware_concurrency falla

    auto worker = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < width; ++x) {
                int baseIndex = (y * width + x) * CHANNELS;
                for (int c = 0; c < CHANNELS; c++) {
                    int pixVal = input[baseIndex + c];
                    int quant = ((pixVal / quantStep) * quantStep) + (quantStep / 2);
                    float lap = 0.0f;
                    for (int j = -half; j <= half; j++) {
                        for (int i = -half; i <= half; i++) {
                            int nx = std::max(0, std::min(x + i, width - 1));
                            int ny = std::max(0, std::min(y + j, height - 1));
                            int nIndex = (ny * width + nx) * CHANNELS + c;
                            lap += input[nIndex] * lapMask[(j + half) * maskSize + (i + half)];
                        }
                    }
                    output[baseIndex + c] = (fabs(lap) > threshold) ?
                        0 : static_cast<unsigned char>(std::min(255, std::max(0, quant)));
                }
            }
        }
        };

    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        int startY = t * rowsPerThread;
        int endY = (t == numThreads - 1) ? height : startY + rowsPerThread;
        threads.emplace_back(worker, startY, endY);
    }

    for (auto& th : threads) {
        th.join();
    }
}


// ---------------------------------------------------------------------
// GPU - Kernel que aplica el filtro Cartoon + Laplaciano unificado
// ---------------------------------------------------------------------
__global__ void cartoonLaplaceKernel(const unsigned char* input,
    unsigned char* output,
    const float* lapMask,
    int width, int height,
    int maskSize,
    int quantStep,
    float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // columna
    int y = blockIdx.y * blockDim.y + threadIdx.y;    // fila
    if (x >= width || y >= height)
        return;

    int half = maskSize / 2;
    // Procesar cada canal (R, G y B)
    for (int c = 0; c < CHANNELS; ++c) {
        int idx = (y * width + x) * CHANNELS + c;

        // 1) Cuantización
        int origVal = input[idx];
        int quant = ((origVal / quantStep) * quantStep) + (quantStep / 2);
        quant = MAX(0, MIN(255, quant));

        // 2) Convolución con el filtro Laplaciano
        float lap = 0.0f;
        for (int j = -half; j <= half; ++j) {
            for (int i = -half; i <= half; ++i) {
                int nx = MIN(MAX(x + i, 0), width - 1);
                int ny = MIN(MAX(y + j, 0), height - 1);
                float w = lapMask[(j + half) * maskSize + (i + half)];
                lap += input[(ny * width + nx) * CHANNELS + c] * w;
            }
        }
        // 3) Evaluación del threshold para determinar borde
        output[idx] = (fabsf(lap) > threshold) ? 0 : static_cast<unsigned char>(MAX(0, MIN(quant, 255)));
    }
}

// ---------------------------------------------------------------------
// Función que procesa la imagen con el filtro para un tamaño de máscara dado
// ---------------------------------------------------------------------
void processCartoonLaplace(const unsigned char* h_input,
    int width, int height,
    int maskSize,
    float sigma,
    int quantStep,
    float threshold,
    const std::string& prefixOut)
{
    // 1. Generar el filtro Laplaciano (LoG)
    float* h_lapMask = new float[maskSize * maskSize];
    generateLoGFilter(h_lapMask, maskSize, sigma);

    size_t imgSize = width * height * CHANNELS * sizeof(unsigned char);

    // Reservar memoria para los resultados (CPU y GPU)
    unsigned char* h_outputCPU = new unsigned char[width * height * CHANNELS];
    unsigned char* h_outputGPU = new unsigned char[width * height * CHANNELS];

    // -------------------- CPU --------------------
    auto t_start = std::chrono::high_resolution_clock::now();
    // Ejecuta la versión paralela de la CPU (ahora utilizando todos los hilos)
    cartoonLaplaceCPU(h_input, h_outputCPU, h_lapMask, width, height, maskSize, quantStep, threshold);
    auto t_end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double>(t_end - t_start).count();

    std::string cpuFile = prefixOut + "_cpu_" + std::to_string(maskSize) + "x" + std::to_string(maskSize) + ".jpg";
    stbi_write_jpg(cpuFile.c_str(), width, height, CHANNELS, h_outputCPU, 100);

    // -------------------- GPU --------------------
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_lapMask = nullptr;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_lapMask, maskSize * maskSize * sizeof(float));

    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lapMask, h_lapMask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    cartoonLaplaceKernel << <grid, block >> > (d_input, d_output, d_lapMask, width, height, maskSize, quantStep, threshold);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float gpuTimeMs = 0.f;
    cudaEventElapsedTime(&gpuTimeMs, startEvent, stopEvent);
    double gpuTime = gpuTimeMs / 1000.0; // Convertir a segundos

    std::string gpuFile = prefixOut + "_gpu_" + std::to_string(maskSize) + "x" + std::to_string(maskSize) + ".jpg";
    cudaMemcpy(h_outputGPU, d_output, imgSize, cudaMemcpyDeviceToHost);
    stbi_write_jpg(gpuFile.c_str(), width, height, CHANNELS, h_outputGPU, 100);

    // Liberar memoria GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lapMask);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Imprimir resultados en consola para la máscara en cuestión
    std::cout << "\nProcesando kernel de tamanio: " << maskSize << "x" << maskSize << "\n";
    std::cout << " Tiempo CPU: " << cpuTime * 1000.0 << " ms \n";
    std::cout << " Tiempo GPU: " << gpuTime * 1000.0 << " ms \n";

    // Comparación de resultados: suma de diferencias absolutas por píxel
    size_t totalPixels = width * height * CHANNELS;
    double errorSum = 0.0;
    for (size_t i = 0; i < totalPixels; i++) {
        errorSum += abs(static_cast<int>(h_outputCPU[i]) - static_cast<int>(h_outputGPU[i]));
    }
    std::cout << std::fixed << std::setprecision(10);
    std::cout << " Error (diferencia de pixeles): " << errorSum << "\n";

    // Liberar memoria CPU
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    delete[] h_lapMask;
}

int main() {
    // Cargar imagen de entrada ("input.jpg") en escala de colores
    int width, height, ch;
    unsigned char* h_input = stbi_load("input.jpg", &width, &height, &ch, CHANNELS);
    if (!h_input) {
        std::cerr << "Error al cargar la imagen input_.jpg\n";
        return -1;
    }
    std::cout << "Imagen cargada: " << width << "x" << height << ", canales: " << CHANNELS << "\n";

    // Parámetros de la operación
    int quantStep = 64;      // Paso de cuantización para el efecto cartoon
    float threshold = 20.0f; // Umbral para detección de bordes
    float sigma = 2.0f;      // Sigma para el filtro LoG

    // Se procesará para tres tamaños de máscara: 9x9, 13x13 y 21x21
    processCartoonLaplace(h_input, width, height, 9, sigma, quantStep, threshold, "cartoonLaplace");
    processCartoonLaplace(h_input, width, height, 13, sigma, quantStep, threshold, "cartoonLaplace");
    processCartoonLaplace(h_input, width, height, 21, sigma, quantStep, threshold, "cartoonLaplace");

    stbi_image_free(h_input);
    return 0;
}
