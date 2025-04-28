import numpy as np
from PIL import Image
import time
import math
import os
import numba
from numba import njit, prange
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Constants
CHANNELS = 3
PI = math.pi

# ---------------------------------------------------------------------
# Generar filtro Laplacian of Gaussian (LoG)
# ---------------------------------------------------------------------
@njit
def generate_log_filter(size, sigma):
    half = size // 2
    mask = np.empty((size, size), dtype=np.float32)
    factor = -1.0 / (PI * sigma ** 4)
    for j in range(-half, half + 1):
        for i in range(-half, half + 1):
            r2 = i * i + j * j
            expo = math.exp(-r2 / (2.0 * sigma * sigma))
            mask[j + half, i + half] = factor * (1.0 - (r2 / (2.0 * sigma * sigma))) * expo
    return mask

# ---------------------------------------------------------------------
# CPU - Cartoon + Laplacian usando Numba paralelo
# ---------------------------------------------------------------------
@njit(parallel=True)
def cartoon_laplace_cpu_numba(input_img, lap_mask, width, height, mask_size, quant_step, threshold):
    output = np.empty_like(input_img)
    half = mask_size // 2
    for y in prange(height):
        for x in range(width):
            base = y * width + x
            for c in range(CHANNELS):
                pix_val = input_img[base * CHANNELS + c]
                quant = ((pix_val // quant_step) * quant_step) + (quant_step // 2)
                if quant < 0:
                    quant = 0
                elif quant > 255:
                    quant = 255
                lap = 0.0
                for j in range(-half, half + 1):
                    for i in range(-half, half + 1):
                        nx = x + i
                        if nx < 0:
                            nx = 0
                        elif nx >= width:
                            nx = width - 1
                        ny = y + j
                        if ny < 0:
                            ny = 0
                        elif ny >= height:
                            ny = height - 1
                        lap += input_img[(ny * width + nx) * CHANNELS + c] * lap_mask[j + half, i + half]
                output[base * CHANNELS + c] = 0 if abs(lap) > threshold else quant
    return output

# ---------------------------------------------------------------------
# CUDA kernel para GPU
# ---------------------------------------------------------------------
kernel_code = """
#define CHANNELS 3
__global__ void cartoonLaplaceKernel(unsigned char* input, unsigned char* output, float* lapMask,
                                     int width, int height, int maskSize, int quantStep, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int half = maskSize / 2;
    int base = (y * width + x) * CHANNELS;
    for (int c = 0; c < CHANNELS; ++c) {
        int idx = base + c;
        int origVal = input[idx];
        int quant = ((origVal / quantStep) * quantStep) + (quantStep / 2);
        quant = max(0, min(255, quant));
        float lap = 0.0f;
        for (int j = -half; j <= half; ++j) {
            for (int i = -half; i <= half; ++i) {
                int nx = min(max(x + i, 0), width - 1);
                int ny = min(max(y + j, 0), height - 1);
                lap += input[(ny * width + nx) * CHANNELS + c] * lapMask[(j + half) * maskSize + (i + half)];
            }
        }
        output[idx] = (fabsf(lap) > threshold) ? 0 : (unsigned char)quant;
    }
}
"""
mod = SourceModule(kernel_code)
cartoon_laplace_gpu = mod.get_function("cartoonLaplaceKernel")

# ---------------------------------------------------------------------
# Función principal de procesamiento
# ---------------------------------------------------------------------
def process_cartoon_laplace(input_arr, width, height, mask_size, sigma, quant_step, threshold, prefix):
    # Generar máscara LoG
    lap_mask = generate_log_filter(mask_size, sigma)

    # CPU con Numba paralelo
    t0 = time.time()
    out_cpu = cartoon_laplace_cpu_numba(input_arr, lap_mask, width, height, mask_size, quant_step, threshold)
    t1 = time.time()

    # Guardar CPU
    cpu_img = Image.fromarray(out_cpu.reshape((height, width, CHANNELS)).astype(np.uint8))
    cpu_img.save(f"{prefix}_cpu_{mask_size}x{mask_size}.jpg")

    # GPU con PyCUDA
    inp = input_arr.astype(np.uint8)
    out_gpu = np.empty_like(inp)
    d_in = cuda.mem_alloc(inp.nbytes)
    d_out = cuda.mem_alloc(out_gpu.nbytes)
    d_mask = cuda.mem_alloc(lap_mask.nbytes)
    cuda.memcpy_htod(d_in, inp)
    cuda.memcpy_htod(d_mask, lap_mask.ravel())

    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16)
    start_e = cuda.Event()
    end_e = cuda.Event()
    start_e.record()
    cartoon_laplace_gpu(d_in, d_out, d_mask, np.int32(width), np.int32(height), np.int32(mask_size), np.int32(quant_step), np.float32(threshold), block=block, grid=grid)
    end_e.record()
    end_e.synchronize()
    t2 = start_e.time_till(end_e) * 1e-3

    cuda.memcpy_dtoh(out_gpu, d_out)
    gpu_img = Image.fromarray(out_gpu.reshape((height, width, CHANNELS)).astype(np.uint8))
    gpu_img.save(f"{prefix}_gpu_{mask_size}x{mask_size}.jpg")

    # Error
    error = np.abs(out_cpu.astype(np.int32) - out_gpu.astype(np.int32)).sum()

    print(f"\nTamaño kernel: {mask_size}x{mask_size}")
    print(f"Tiempo CPU: {(t1-t0)*1000:.2f} ms")
    print(f"Tiempo GPU: {t2*1000:.2f} ms")
    print(f"Error (suma de diferencias): {error}")

    # Liberar GPU
    d_in.free(); d_out.free(); d_mask.free()

if __name__ == '__main__':
    # Cargar y aplanar imagen
    img = Image.open('input.jpg').convert('RGB')
    w, h = img.size
    arr = np.array(img, dtype=np.uint8).ravel()

    quant_step = 64
    threshold = 20.0
    sigma = 2.0
    for size in [9, 13, 21]:
        process_cartoon_laplace(arr, w, h, size, sigma, quant_step, threshold, 'cartoonLaplace')
