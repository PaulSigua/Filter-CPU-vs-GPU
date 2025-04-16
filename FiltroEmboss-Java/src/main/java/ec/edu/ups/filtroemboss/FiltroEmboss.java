/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package ec.edu.ups.filtroemboss;

/**
 *
 * @author andy
 */
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import java.util.concurrent.*;

public class FiltroEmboss {

    // Método para crear el kernel de embosse
    public static double[] crearKernelEmboss(int kernelSize) {
        double[] kernel = new double[kernelSize * kernelSize];
        int half = kernelSize / 2;
        for (int y = 0; y < kernelSize; y++) {
            for (int x = 0; x < kernelSize; x++) {
                if (x < half && y < half)
                    kernel[y * kernelSize + x] = -1;
                else if (x > half && y > half || (x == half && y == half))
                    kernel[y * kernelSize + x] = 1;
            }
        }
        return kernel;
    }

    // Método para aplicar el filtro de forma secuencial
    public static double[] aplicarConvolucionSecuencial(BufferedImage image, double[] kernel, int width, int height, int kernelSize) {
        int halfKernel = kernelSize / 2;
        double[] resultado = new double[width * height];

        for (int y = halfKernel; y < height - halfKernel; y++) {
            for (int x = halfKernel; x < width - halfKernel; x++) {
                double suma = 0.0;
                for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int pixel = image.getRGB(x + kx, y + ky) & 0xFF; // Obtener el valor de gris del píxel
                        suma += pixel * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                    }
                }
                resultado[y * width + x] = suma;
            }
        }
        return resultado;
    }

    // Método para aplicar el filtro de forma paralela
    public static double[] aplicarConvolucionParalela(BufferedImage image, double[] kernel, int width, int height, int kernelSize) throws InterruptedException, ExecutionException {
        int halfKernel = kernelSize / 2;
        double[] resultado = new double[width * height];

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<Void>> futures = new ArrayList<>();

        for (int y = halfKernel; y < height - halfKernel; y++) {
            final int fy = y;
            futures.add(executor.submit(() -> {
                for (int x = halfKernel; x < width - halfKernel; x++) {
                    double suma = 0.0;
                    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                            int pixel = image.getRGB(x + kx, fy + ky) & 0xFF;
                            suma += pixel * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                        }
                    }
                    resultado[fy * width + x] = suma;
                }
                return null;
            }));
        }

        for (Future<Void> future : futures) {
            future.get();
        }

        executor.shutdown();
        return resultado;
    }

    // Método para guardar la imagen resultante
    public static void guardarImagen(double[] resultado, int width, int height, String filename) throws IOException {
        BufferedImage salida = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;

        for (double pixel : resultado) {
            if (pixel < min) min = pixel;
            if (pixel > max) max = pixel;
        }

        for (int i = 0; i < width * height; i++) {
            int pixel = (int) ((resultado[i] - min) / (max - min) * 255);
            pixel = Math.max(0, Math.min(255, pixel));
            salida.setRGB(i % width, i / width, (pixel << 16) | (pixel << 8) | pixel);
        }

        ImageIO.write(salida, "jpg", new File(filename));
    }

    // Método para calcular el error porcentual entre dos resultados
    public static double calcularErrorPorcentual(double[] resultado1, double[] resultado2) {
        int diferentes = 0;
        double tolerancia = 1e-3;

        for (int i = 0; i < resultado1.length; i++) {
            if (Math.abs(resultado1[i] - resultado2[i]) > tolerancia) {
                diferentes++;
            }
        }

        return 100.0 * diferentes / resultado1.length;
    }

    // Método principal
    // Método principal
public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
    BufferedImage imagen = ImageIO.read(new File("image2.jpg"));
    int width = imagen.getWidth();
    int height = imagen.getHeight();

    int[] kernelSizes = {9, 13, 21};

    for (int k : kernelSizes) {
        double[] kernel = crearKernelEmboss(k);

        System.out.println("------------------------------------------------");
        System.out.println("Kernel de tamaño " + k + "x" + k);

        // Secuencial
        long startTime = System.nanoTime();
        double[] resultadoSecuencial = aplicarConvolucionSecuencial(imagen, kernel, width, height, k);
        long endTime = System.nanoTime();
        System.out.printf("Tiempo secuencial: %.6f milisegundos\n", (endTime - startTime) / 1e6);

        // Paralelo
        startTime = System.nanoTime();
        double[] resultadoParalelo = aplicarConvolucionParalela(imagen, kernel, width, height, k);
        endTime = System.nanoTime();
        System.out.printf("Tiempo paralelo:   %.6f milisegundos\n", (endTime - startTime) / 1e6);

        // Error
        double error = calcularErrorPorcentual(resultadoSecuencial, resultadoParalelo);
        System.out.printf("Error (pixeles diferentes): %.8f %%\n", error);

        // Guardar imágenes
        guardarImagen(resultadoSecuencial, width, height, "resultado_secuencial_" + k + ".jpg");
        guardarImagen(resultadoParalelo, width, height, "resultado_paralelo_" + k + ".jpg");
    }
}

}