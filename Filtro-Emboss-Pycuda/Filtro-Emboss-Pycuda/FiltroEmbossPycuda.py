import numpy as np
import pycuda.driver as cuda # type: ignore
import pycuda.autoinit # type: ignore
from pycuda.compiler import SourceModule # type: ignore
from PIL import Image
from numba import njit, prange # type: ignore

import time

def create_emboss_kernel(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    half = kernel_size // 2
    for y in range(kernel_size):
        for x in range(kernel_size):
            if x < half and y < half:
                kernel[y, x] = -1
            elif (x > half and y > half) or (x == half and y == half):
                kernel[y, x] = 1
    return kernel

@njit(parallel=True)
def apply_convolution_cpu(image, kernel):
    height, width = image.shape
    k_size = kernel.shape[0]
    half = k_size // 2
    result = np.zeros((height, width), dtype=np.float64)
    for y in prange(half, height - half):
        for x in range(half, width - half):
            sum_val = 0.0
            for ky in range(-half, half + 1):
                for kx in range(-half, half + 1):
                    pixel = image[y + ky, x + kx]
                    weight = kernel[ky + half, kx + half]
                    sum_val += pixel * weight
            result[y, x] = sum_val
    return result

def normalize_and_save(image, filename):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = ((image - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(norm_img)
    img.save(filename)

def run_convolution_comparison(image, kernel_size):
    print(f"\n--- Kernel Size: {kernel_size}x{kernel_size} ---")
    kernel = create_emboss_kernel(kernel_size)
    
    # CPU
    start_cpu = time.time()
    result_cpu = apply_convolution_cpu(image, kernel)
    end_cpu = time.time()
    print(f"CPU Time: {(end_cpu - start_cpu) * 1000:.4f} ms")
    normalize_and_save(result_cpu, f"resultado_cpu_{kernel_size}x{kernel_size}.jpg")
    
    # GPU
    img_height, img_width = image.shape
    img_bytes = image.nbytes
    kernel_bytes = kernel.nbytes
    result_bytes = result_cpu.nbytes

    d_image = cuda.mem_alloc(img_bytes)
    d_kernel = cuda.mem_alloc(kernel_bytes)
    d_result = cuda.mem_alloc(result_bytes)

    cuda.memcpy_htod(d_image, image)
    cuda.memcpy_htod(d_kernel, kernel)

    mod = SourceModule(f"""
    __global__ void applyConvolutionGPU(unsigned char* image, double* kernel, double* result, int width, int height, int ksize)
    {{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int half = ksize / 2;
        
        if (x >= half && x < width - half && y >= half && y < height - half)
        {{
            double sum = 0.0;
            for (int ky = -half; ky <= half; ky++)
            {{
                for (int kx = -half; kx <= half; kx++)
                {{
                    int img_idx = (y + ky) * width + (x + kx);
                    int ker_idx = (ky + half) * ksize + (kx + half);
                    sum += image[img_idx] * kernel[ker_idx];
                }}
            }}
            result[y * width + x] = sum;
        }}
    }}
    """)
    
    func = mod.get_function("applyConvolutionGPU")
    
    block = (16, 16, 1)
    grid = ((img_width + block[0] - 1) // block[0], (img_height + block[1] - 1) // block[1])

    start_gpu = cuda.Event()
    end_gpu = cuda.Event()
    start_gpu.record()
    
    func(d_image, d_kernel, d_result, np.int32(img_width), np.int32(img_height), np.int32(kernel_size), block=block, grid=grid)
    
    end_gpu.record()
    end_gpu.synchronize()
    elapsed_time = start_gpu.time_till(end_gpu)
    print(f"GPU Time: {elapsed_time:.4f} ms")

    result_gpu = np.empty_like(result_cpu)
    cuda.memcpy_dtoh(result_gpu, d_result)

    normalize_and_save(result_gpu, f"resultado_gpu_{kernel_size}x{kernel_size}.jpg")

    # Comparación
    tolerance = 1e-3
    diff_pixels = np.sum(np.abs(result_cpu - result_gpu) > tolerance)
    error_percentage = 100.0 * diff_pixels / (img_width * img_height)
    print(f"Error (diferencia de píxeles): {error_percentage:.8f} %")

def main():
    img = Image.open('image2.jpg').convert('L')
    image = np.array(img)

    kernel_sizes = [9, 13, 21]
    for ksize in kernel_sizes:
        run_convolution_comparison(image, ksize)

if __name__ == "__main__":
    main()
