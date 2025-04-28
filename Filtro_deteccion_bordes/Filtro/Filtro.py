import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
from numba import njit, prange

# Función para generar el kernel
def generar_kernel_bordes(size):
    kernel = np.ones((size, size), dtype=np.float32)
    center = size // 2
    total = size * size
    kernel[center, center] = -1.0 * (total - 1)
    return kernel


# Convolución CPU optimizada con Numba (paralelizada)
@njit #(parallel=True)
def ejecutar_cpu(image, kernel):
    height, width, channels = image.shape
    k_size = kernel.shape[0]
    k_half = k_size // 2
    output = np.empty_like(image)

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                sum = 0.0
                for ky in range(-k_half, k_half + 1):
                    for kx in range(-k_half, k_half + 1):
                        px = x + kx
                        if px < 0: px = 0
                        elif px >= width: px = width - 1
                        py = y + ky
                        if py < 0: py = 0
                        elif py >= height: py = height - 1
                        sum += image[py, px, c] * kernel[ky + k_half, kx + k_half]
                val = int(sum)
                if val < 0: val = 0
                elif val > 255: val = 255
                output[y, x, c] = val
    return output

# CUDA kernel (convolución en GPU)
cuda_source = """
__global__ void convolucion_gpu(
    unsigned char *input, unsigned char *output,
    int width, int height, int channels,
    float *kernel, int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k2 = ksize / 2;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -k2; ky <= k2; ++ky) {
            for (int kx = -k2; kx <= k2; ++kx) {
                int xx = min(max(x + kx, 0), width - 1);
                int yy = min(max(y + ky, 0), height - 1);
                int img_idx = (yy * width + xx) * channels + c;
                int k_idx = (ky + k2) * ksize + (kx + k2);
                sum += input[img_idx] * kernel[k_idx];
            }
        }
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = min(max(int(sum), 0), 255);
    }
}
"""

mod = SourceModule(cuda_source)
convolucion_gpu = mod.get_function("convolucion_gpu")

#Convolución en GPU
def ejecutar_gpu(input_img, kernel):
    h, w, c = input_img.shape
    ksize = kernel.shape[0]

    input_flat = input_img.ravel().astype(np.uint8)
    output_flat = np.empty_like(input_flat)
    kernel_flat = kernel.ravel().astype(np.float32)

    # Asignar memoria en el dispositivo
    d_input = drv.mem_alloc(input_flat.nbytes)
    d_output = drv.mem_alloc(output_flat.nbytes)
    d_kernel = drv.mem_alloc(kernel_flat.nbytes)

    # Copiar datos al dispositivo
    drv.memcpy_htod(d_input, input_flat)
    drv.memcpy_htod(d_kernel, kernel_flat)

    block = (16, 16, 1)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)

    start = time.time()
    convolucion_gpu(d_input, d_output, np.int32(w), np.int32(h), np.int32(c), 
                    d_kernel, np.int32(ksize), block=block, grid=grid)
    drv.Context.synchronize()
    gpu_time = (time.time() - start) * 1000
    print(f"GPU tiempo ({ksize}x{ksize}): {gpu_time:.2f} ms")

    drv.memcpy_dtoh(output_flat, d_output)
    return output_flat.reshape((h, w, c)).astype(np.uint8)

# Función principal
def main():
    # Cargar la imagen con OpenCV
    img = cv2.imread("image2.jpg")
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = input_img.shape
    print(f"Imagen cargada: {w}x{h}, Canales: {c}")

    kernel_sizes = [9, 13, 21]  # Tamaño del kernel
    for k in kernel_sizes:
        print(f"\nProcesando kernel {k}x{k}")
        kernel = generar_kernel_bordes(k)

        # Convolución en CPU
        print("Procesamiento CPU")
        start = time.time()
        cpu_out = ejecutar_cpu(input_img, kernel)
        cpu_time = (time.time() - start) * 1000
        print(f"CPU tiempo ({k}x{k}): {cpu_time:.2f} ms")
        cv2.imwrite(f"cpu_{k}x{k}.jpg", cv2.cvtColor(cpu_out, cv2.COLOR_RGB2BGR))

        # Convolución en GPU
        print("Procesamiento GPU")
        gpu_out = ejecutar_gpu(input_img, kernel)
        cv2.imwrite(f"gpu_{k}x{k}.jpg", cv2.cvtColor(gpu_out, cv2.COLOR_RGB2BGR))

        # Comparar las salidas
        diff = np.any(cpu_out != gpu_out, axis=2)
        errors = np.count_nonzero(diff)
        pct = 100.0 * errors / (w * h)
        print(f"Diferencias píxel {k}x{k}: {errors} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
