package ec.edu.ups.java.proyect.filter;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;

/**
 *
 * @author mateo
 */
public class JavaProyectFilter {

    private static final int quantStep = 64;
    private static final float threshold = 20.0f;
    private static final float sigma = 2.0f;
    private static final int CHANNELS = 3; // procesamos R, G y B

    static class Result {
        int maskSize;
        double timeSeq;
        double timePar;
        int error; // Cantidad de píxeles diferentes entre CPU y Paralelo

        public Result(int maskSize, double timeSeq, double timePar, int error) {
            this.maskSize = maskSize;
            this.timeSeq = timeSeq;
            this.timePar = timePar;
            this.error = error;
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                File inputFile = seleccionarImagen();
                if (inputFile == null) {
                    System.err.println("No se seleccionó una imagen.");
                    return;
                }

                BufferedImage inputImg = ImageIO.read(inputFile);
                if (inputImg == null) {
                    System.err.println("Error al cargar la imagen: " + inputFile.getAbsolutePath());
                    return;
                }
                // Convertir la imagen al formato de 3 bytes por píxel (color) si no lo está
                BufferedImage colorImg = convertTo3ByteColor(inputImg);
                int width = colorImg.getWidth();
                int height = colorImg.getHeight();
                System.out.println("Imagen cargada: " + width + "x" + height);
                // Ahora el arreglo tendrá width * height * CHANNELS elementos
                byte[] inputArray = imageToByteArrayRGB(colorImg);

                int[] maskSizes = {9, 13, 21};
                List<Result> results = new ArrayList<>();

                for (int maskSize : maskSizes) {
                    float[] lapMask = new float[maskSize * maskSize];
                    generateLoGFilter(lapMask, maskSize, sigma);
                    // Los arreglos de salida serán para 3 canales
                    byte[] outputCPU = new byte[width * height * CHANNELS];
                    byte[] outputPar = new byte[width * height * CHANNELS];

                    // -------------------- Procesamiento Secuencial --------------------
                    long startSeq = System.nanoTime();
                    cartoonLaplaceCPURGB(inputArray, outputCPU, lapMask, width, height, maskSize, quantStep, threshold);
                    long endSeq = System.nanoTime();
                    double timeSeq = (endSeq - startSeq) / 1_000_000.0;

                    BufferedImage outCPU = byteArrayToImageRGB(outputCPU, width, height);
                    ImageIO.write(outCPU, "jpg", new File("cartoon_sec_" + maskSize + ".jpg"));

                    // -------------------- Procesamiento Paralelo --------------------
                    long startPar = System.nanoTime();
                    cartoonLaplaceParallelRGB(inputArray, outputPar, lapMask, width, height, maskSize, quantStep, threshold);
                    long endPar = System.nanoTime();
                    double timePar = (endPar - startPar) / 1_000_000.0;

                    BufferedImage outPar = byteArrayToImageRGB(outputPar, width, height);
                    ImageIO.write(outPar, "jpg", new File("cartoon_par_" + maskSize + ".jpg"));

                    // Calcular el error: cantidad de píxeles diferentes entre CPU y Paralelo
                    int error = calculateError(outputCPU, outputPar, CHANNELS);

                    System.out.printf("\nMascara %dx%d:\nCPU Secuencial: %.2f ms\nCPU Paralelo: %.2f ms\nError (pixeles diferentes): %.2f\n",
                            maskSize, maskSize,
                            timeSeq, timePar,
                            (double) error);
                    
                    results.add(new Result(maskSize, timeSeq, timePar, error));
                }

                // Mostrar gráfica comparativa de resultados (tiempos y error)
                new GraphFrame(results).setVisible(true);

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * Calcula el error entre dos arreglos de píxeles (por ejemplo, de las imágenes de CPU y GPU).
     * Se recorre cada píxel (considerando sus 3 canales) y si para ese píxel hay al menos un canal diferente,
     * se incrementa el error.
     *
     * @param cpuResult Arreglo resultante del procesamiento secuencial.
     * @param gpuResult Arreglo resultante del procesamiento paralelo.
     * @param channels Número de canales por píxel.
     * @return Cantidad de píxeles diferentes.
     */
    public static int calculateError(byte[] cpuResult, byte[] gpuResult, int channels) {
        int error = 0;
        int numPixels = cpuResult.length / channels;
        for (int i = 0; i < numPixels; i++) {
            boolean diff = false;
            for (int c = 0; c < channels; c++) {
                if (cpuResult[i * channels + c] != gpuResult[i * channels + c]) {
                    diff = true;
                    break;
                }
            }
            if (diff) {
                error++;
            }
        }
        return error;
    }

    public static File seleccionarImagen() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileFilter(new FileNameExtensionFilter("Imágenes", "jpg", "jpeg", "png"));
        return chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION ? chooser.getSelectedFile() : null;
    }

    /**
     * Convierte una imagen a un formato de 3 bytes por píxel (color), de tipo TYPE_3BYTE_BGR.
     */
    public static BufferedImage convertTo3ByteColor(BufferedImage original) {
        if (original.getType() == BufferedImage.TYPE_3BYTE_BGR) {
            return original;
        }
        BufferedImage colorImage = new BufferedImage(original.getWidth(), original.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        Graphics g = colorImage.getGraphics();
        g.drawImage(original, 0, 0, null);
        g.dispose();
        return colorImage;
    }

    /**
     * Convierte un BufferedImage en un arreglo de bytes para imágenes RGB.
     */
    public static byte[] imageToByteArrayRGB(BufferedImage img) {
        // Se asume que la imagen es de tipo TYPE_3BYTE_BGR.
        DataBufferByte db = (DataBufferByte) img.getRaster().getDataBuffer();
        byte[] data = db.getData();
        byte[] result = new byte[data.length];
        System.arraycopy(data, 0, result, 0, data.length);
        return result;
    }

    /**
     * Reconstruye una imagen RGB a partir de un arreglo de bytes.
     */
    public static BufferedImage byteArrayToImageRGB(byte[] data, int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        img.getRaster().setDataElements(0, 0, width, height, data);
        return img;
    }

    /**
     * Genera la máscara LoG (Laplacian of Gaussian) para un tamaño dado.
     */
    public static void generateLoGFilter(float[] mask, int size, float sigma) {
        int half = size / 2;
        for (int j = -half; j <= half; j++) {
            for (int i = -half; i <= half; i++) {
                float r2 = i * i + j * j;
                float factor = -1f / ((float) Math.PI * (float) Math.pow(sigma, 4));
                float expo = (float) Math.exp(-r2 / (2 * sigma * sigma));
                float value = factor * (1 - (r2 / (2 * sigma * sigma))) * expo;
                mask[(j + half) * size + (i + half)] = value;
            }
        }
    }

    /**
     * Versión secuencial del filtro para imágenes en color (RGB).
     */
    public static void cartoonLaplaceCPURGB(byte[] input, byte[] output, float[] lapMask,
                                             int width, int height, int maskSize,
                                             int quantStep, float threshold) {
        int half = maskSize / 2;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int baseIndex = (y * width + x) * CHANNELS;
                for (int c = 0; c < CHANNELS; c++) {
                    int pixVal = Byte.toUnsignedInt(input[baseIndex + c]);
                    int quant = ((pixVal / quantStep) * quantStep) + (quantStep / 2);
                    float lap = 0;
                    for (int j = -half; j <= half; j++) {
                        for (int i = -half; i <= half; i++) {
                            int nx = Math.max(0, Math.min(x + i, width - 1));
                            int ny = Math.max(0, Math.min(y + j, height - 1));
                            int nIndex = (ny * width + nx) * CHANNELS + c;
                            lap += Byte.toUnsignedInt(input[nIndex]) * lapMask[(j + half) * maskSize + (i + half)];
                        }
                    }
                    output[baseIndex + c] = (Math.abs(lap) > threshold) ? 0 : (byte) Math.min(255, Math.max(0, quant));
                }
            }
        }
    }

    /**
     * Versión paralela del filtro para imágenes en color (RGB).
     */
    public static void cartoonLaplaceParallelRGB(byte[] input, byte[] output, float[] lapMask,
                                                  int width, int height, int maskSize,
                                                  int quantStep, float threshold) {
        int threads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        int rowsPerThread = height / threads;
        int half = maskSize / 2;

        for (int t = 0; t < threads; t++) {
            int startY = t * rowsPerThread;
            int endY = (t == threads - 1) ? height : startY + rowsPerThread;
            executor.submit(() -> {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < width; x++) {
                        int baseIndex = (y * width + x) * CHANNELS;
                        for (int c = 0; c < CHANNELS; c++) {
                            int pixVal = Byte.toUnsignedInt(input[baseIndex + c]);
                            int quant = ((pixVal / quantStep) * quantStep) + (quantStep / 2);
                            float lap = 0;
                            for (int j = -half; j <= half; j++) {
                                for (int i = -half; i <= half; i++) {
                                    int nx = Math.max(0, Math.min(x + i, width - 1));
                                    int ny = Math.max(0, Math.min(y + j, height - 1));
                                    int nIndex = (ny * width + nx) * CHANNELS + c;
                                    lap += Byte.toUnsignedInt(input[nIndex]) * lapMask[(j + half) * maskSize + (i + half)];
                                }
                            }
                            output[baseIndex + c] = (Math.abs(lap) > threshold) ? 0 : (byte) Math.min(255, Math.max(0, quant));
                        }
                    }
                }
            });
        }
        executor.shutdown();
        try {
            executor.awaitTermination(1, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

class GraphFrame extends JFrame {
    public GraphFrame(List<JavaProyectFilter.Result> results) {
        setTitle("Comparación Secuencial vs Paralelo");
        setSize(700, 500);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        add(new GraphPanel(results));
    }
}

class GraphPanel extends JPanel {
    private final List<JavaProyectFilter.Result> results;

    public GraphPanel(List<JavaProyectFilter.Result> results) {
        this.results = results;
        setPreferredSize(new Dimension(700, 500));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        int panelWidth = getWidth();
        int panelHeight = getHeight();
        int margin = 60;
        int barWidth = 40;
        int gap = 40;

        // Buscar el tiempo máximo para escalar las barras
        double maxTime = results.stream().mapToDouble(r -> Math.max(r.timeSeq, r.timePar)).max().orElse(100);

        g.setFont(new Font("Arial", Font.BOLD, 14));
        g.drawString("Tiempo de ejecución (ms)", panelWidth / 2 - 80, margin / 2);

        for (int i = 0; i < results.size(); i++) {
            JavaProyectFilter.Result r = results.get(i);
            int xBase = margin + i * (2 * barWidth + gap);
            int heightSeq = (int) ((r.timeSeq / maxTime) * (panelHeight - 2 * margin));
            int heightPar = (int) ((r.timePar / maxTime) * (panelHeight - 2 * margin));

            // Dibujar barra secuencial
            g.setColor(Color.BLUE);
            g.fillRect(xBase, panelHeight - margin - heightSeq, barWidth, heightSeq);
            g.setColor(Color.BLACK);
            g.drawRect(xBase, panelHeight - margin - heightSeq, barWidth, heightSeq);
            g.drawString("S", xBase + 10, panelHeight - margin + 15);
            g.drawString(String.format("%.1f", r.timeSeq), xBase, panelHeight - margin - heightSeq - 5);

            // Dibujar barra paralela
            g.setColor(Color.GREEN);
            g.fillRect(xBase + barWidth, panelHeight - margin - heightPar, barWidth, heightPar);
            g.setColor(Color.BLACK);
            g.drawRect(xBase + barWidth, panelHeight - margin - heightPar, barWidth, heightPar);
            g.drawString("P", xBase + barWidth + 10, panelHeight - margin + 15);
            g.drawString(String.format("%.1f", r.timePar), xBase + barWidth, panelHeight - margin - heightPar - 5);

            // Etiqueta del tamaño de la máscara
            g.drawString(r.maskSize + "x" + r.maskSize, xBase + 5, panelHeight - margin + 35);
            // Mostrar el error en cada kernel con dos decimales
            g.setColor(Color.RED);
            g.drawString("Error: " + String.format("%.2f", (double)r.error), xBase, panelHeight - margin + 55);
        }
    }
}