/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.filtrobordes;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/**
 *
 * @author jeiso
 */
public class FiltroBordes {

    // Función para generar una máscara de detección de bordes (tipo Laplaciano ampliado)
    public static int[][] generarMascaraBordes(int tamaño) {
        int[][] mascara = new int[tamaño][tamaño];
        int centro = tamaño / 2;

        for (int i = 0; i < tamaño; i++) {
            for (int j = 0; j < tamaño; j++) {
                mascara[i][j] = -1;
            }
        }

        mascara[centro][centro] = tamaño * tamaño - 1;
        return mascara;
    }

    // Filtro secuencial
    public static BufferedImage aplicarFiltroBiordes(BufferedImage imagen, int[][] mascara) {
        int ancho = imagen.getWidth();
        int alto = imagen.getHeight();
        int tamaño = mascara.length;
        int offset = tamaño / 2;

        BufferedImage resultado = new BufferedImage(ancho, alto, BufferedImage.TYPE_INT_RGB);

        for (int y = offset; y < alto - offset; y++) {
            for (int x = offset; x < ancho - offset; x++) {
                int sumaR = 0, sumaG = 0, sumaB = 0;

                for (int i = 0; i < tamaño; i++) {
                    for (int j = 0; j < tamaño; j++) {
                        int pixel = imagen.getRGB(x + j - offset, y + i - offset);
                        int r = (pixel >> 16) & 0xFF;
                        int g = (pixel >> 8) & 0xFF;
                        int b = pixel & 0xFF;

                        sumaR += r * mascara[i][j];
                        sumaG += g * mascara[i][j];
                        sumaB += b * mascara[i][j];
                    }
                }

                sumaR = Math.min(255, Math.max(0, Math.abs(sumaR)));
                sumaG = Math.min(255, Math.max(0, Math.abs(sumaG)));
                sumaB = Math.min(255, Math.max(0, Math.abs(sumaB)));

                int nuevoColor = (0xFF << 24) | (sumaR << 16) | (sumaG << 8) | sumaB;
                resultado.setRGB(x, y, nuevoColor);
            }
        }
        return resultado;
    }

    // Hilo para aplicar filtro en paralelo
    static class FiltroBordesParalelo extends Thread {

        private BufferedImage original;
        private BufferedImage destino;
        private int[][] mascara;
        private int yInicio, yFin;

        public FiltroBordesParalelo(BufferedImage original, BufferedImage destino, int[][] mascara, int yInicio, int yFin) {
            this.original = original;
            this.destino = destino;
            this.mascara = mascara;
            this.yInicio = yInicio;
            this.yFin = yFin;
        }

        @Override
        public void run() {
            int tamaño = mascara.length;
            int offset = tamaño / 2;
            int width = original.getWidth();
            int height = original.getHeight();

            for (int y = Math.max(offset, yInicio); y < Math.min(yFin, height - offset); y++) {
                for (int x = offset; x < width - offset; x++) {
                    int sumaR = 0, sumaG = 0, sumaB = 0;

                    for (int i = 0; i < tamaño; i++) {
                        for (int j = 0; j < tamaño; j++) {
                            int px = original.getRGB(x + j - offset, y + i - offset);
                            int r = (px >> 16) & 0xFF;
                            int g = (px >> 8) & 0xFF;
                            int b = px & 0xFF;

                            sumaR += r * mascara[i][j];
                            sumaG += g * mascara[i][j];
                            sumaB += b * mascara[i][j];
                        }
                    }

                    sumaR = Math.min(255, Math.max(0, Math.abs(sumaR)));
                    sumaG = Math.min(255, Math.max(0, Math.abs(sumaG)));
                    sumaB = Math.min(255, Math.max(0, Math.abs(sumaB)));

                    int nuevoColor = (0xFF << 24) | (sumaR << 16) | (sumaG << 8) | sumaB;
                    destino.setRGB(x, y, nuevoColor);
                }
            }
        }
    }

    // Aplicar filtro en paralelo
    public static BufferedImage aplicarFiltroParalelo(BufferedImage imagen, int[][] mascara, int numThreads) {
        int width = imagen.getWidth();
        int height = imagen.getHeight();
        BufferedImage resultado = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        FiltroBordesParalelo[] hilos = new FiltroBordesParalelo[numThreads];
        int bloqueAltura = height / numThreads;

        for (int i = 0; i < numThreads; i++) {
            int yInicio = i * bloqueAltura;
            int yFin = (i == numThreads - 1) ? height : yInicio + bloqueAltura;

            hilos[i] = new FiltroBordesParalelo(imagen, resultado, mascara, yInicio, yFin);
            hilos[i].start();
        }

        for (int i = 0; i < numThreads; i++) {
            try {
                hilos[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return resultado;
    }

    public static int compararImagenes(BufferedImage img1, BufferedImage img2) {
        int ancho = img1.getWidth();
        int alto = img1.getHeight();
        int diferencias = 0;

        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                if (img1.getRGB(x, y) != img2.getRGB(x, y)) {
                    diferencias++;
                }
            }
        }

        return diferencias;
    }

    public static double calcularPorcentajeError(BufferedImage img1, BufferedImage img2) {
        int totalPixeles = img1.getWidth() * img1.getHeight();
        int diferencias = compararImagenes(img1, img2);
        return (diferencias * 100.0) / totalPixeles;
    }

    // Main
    public static void main(String[] args) throws Exception {
        BufferedImage imagenOriginal = ImageIO.read(new File("C:\\Users\\jeiso\\OneDrive\\Documents\\NetBeansProjects\\FiltroBordes\\src\\main\\java\\com\\mycompany\\filtrobordes\\image2.jpg"));

        String rutaSalida = "C:\\Users\\jeiso\\OneDrive\\Documents\\NetBeansProjects\\FiltroBordes\\Resultados\\";
        int[] tamanios = {9, 13, 21}; 

        for (int tam : tamanios) {
            System.out.println("=========== MÁSCARA " + tam + "x" + tam + " ===========");

            int[][] mascara = generarMascaraBordes(tam);

            // Filtro secuencial
            long inicio = System.currentTimeMillis();
            BufferedImage imagenSecuencial = aplicarFiltroBiordes(imagenOriginal, mascara);
            long fin = System.currentTimeMillis();
            System.out.println("Tiempo Secuencial " + tam + "x" + tam + ": " + (fin - inicio) + " ms");

            // Filtro paralelo
            inicio = System.currentTimeMillis();
            BufferedImage imagenParalela = aplicarFiltroParalelo(imagenOriginal, mascara, 4);
            fin = System.currentTimeMillis();
            System.out.println("Tiempo Paralelo " + tam + "x" + tam + ": " + (fin - inicio) + " ms");

            // Comparación y error
            int diferencias = compararImagenes(imagenSecuencial, imagenParalela);
            double porcentajeError = calcularPorcentajeError(imagenSecuencial, imagenParalela);
            System.out.println("Diferencias encontradas: " + diferencias);
            System.out.printf("Porcentaje de error: %.2f%%\n", porcentajeError);

            // Guardar resultados
            ImageIO.write(imagenSecuencial, "png", new File(rutaSalida + "Secuencial_" + tam + "x" + tam + ".png"));
            ImageIO.write(imagenParalela, "png", new File(rutaSalida + "Paralelo_" + tam + "x" + tam + ".png"));

            System.out.println("Resultados guardados para " + tam + "x" + tam + "\n");
        }

        System.out.println("Todas las imágenes han sido procesadas y guardadas exitosamente.");
    }
}
