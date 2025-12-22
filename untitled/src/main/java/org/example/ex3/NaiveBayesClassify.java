package org.example.ex3;

import java.io.*;
import java.util.*;

public class NaiveBayesClassify {

    static class DataPoint {
        int[] features;
        String label;

        public DataPoint(int a, int b, int c, String label) {
            this.features = new int[]{a, b, c};
            this.label = label;
        }

        @Override
        public String toString() {
            return String.format("A=%d, B=%d, C=%d -> %s",
                    features[0], features[1], features[2], label);
        }
    }

    static class ConditionalProbabilities {
        double priorPlus;   // P(+)
        double priorMinus;  // P(-)

        // P(A=1|+), P(B=1|+), P(C=1|+)
        double[] probGivenPlus = new double[3];

        // P(A=1|-), P(B=1|-), P(C=1|-)
        double[] probGivenMinus = new double[3];

        public void print() {
            System.out.println("\n" + "=".repeat(70));
            System.out.println("Xác suất có điều kiện");
            System.out.println("=".repeat(70));

            System.out.printf("\nXác suất tiên nghiệm:\n");
            System.out.printf("  P(+) = %.4f\n", priorPlus);
            System.out.printf("  P(-) = %.4f\n", priorMinus);

            System.out.printf("\nXác suất có điều kiện cho lớp '+':\n");
            System.out.printf("  P(A=1|+) = %.4f\n", probGivenPlus[0]);
            System.out.printf("  P(B=1|+) = %.4f\n", probGivenPlus[1]);
            System.out.printf("  P(C=1|+) = %.4f\n", probGivenPlus[2]);

            System.out.printf("\nXác suất có điều kiện cho lớp '-':\n");
            System.out.printf("  P(A=1|-) = %.4f\n", probGivenMinus[0]);
            System.out.printf("  P(B=1|-) = %.4f\n", probGivenMinus[1]);
            System.out.printf("  P(C=1|-) = %.4f\n", probGivenMinus[2]);
        }
    }

    public static void main(String[] args) {
        try {
            List<DataPoint> trainingData = readTrainingData("training");
            System.out.println("Cho bộ dữ liệu training có " + trainingData.size() + " điểm training");
            System.out.println("\nTraining Dataset:");
            for (DataPoint p : trainingData) {
                System.out.printf("  %d  %d  %d    %s\n",
                        p.features[0], p.features[1], p.features[2], p.label);
            }

            // Ước lượng các xác suất có điều kiện
            System.out.println("\n" + "=".repeat(70));
            System.out.println("Ước lượng các xác suất có điều kiện");
            System.out.println("=".repeat(70));

            ConditionalProbabilities probs = estimateProbabilities(trainingData);
            probs.print();

            // Phân lớp điểm test
            System.out.println("\n\n" + "=".repeat(70));
            System.out.println("Phân lớp điểm test");
            System.out.println("=".repeat(70));

            List<DataPoint> testData = readTestingData("testing");
            System.out.println("\nCho bộ dữ liệu testing có " + testData.size() + " điểm testing");

            for (DataPoint testPoint : testData) {
                System.out.println("\n" + "-".repeat(70));
                classifyWithNaiveBayes(testPoint, probs);
            }

        } catch (IOException e) {
            System.err.println("Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = NaiveBayesClassify.class.getClassLoader()
                .getResourceAsStream("ex3/" + filename);

        if (is != null) {
            return new BufferedReader(new InputStreamReader(is));
        }
        throw new FileNotFoundException("Không tìm thấy file: " + filename);
    }

    public static List<DataPoint> readTrainingData(String filename) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            String[] parts = line.split("\\s+");
            if (parts.length >= 4) {
                int a = Integer.parseInt(parts[0]);
                int b = Integer.parseInt(parts[1]);
                int c = Integer.parseInt(parts[2]);
                String label = parts[3];
                data.add(new DataPoint(a, b, c, label));
            }
        }

        reader.close();
        return data;
    }

    public static List<DataPoint> readTestingData(String filename) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            String[] parts = line.split("\\s+");
            if (parts.length >= 3) {
                int a = Integer.parseInt(parts[0]);
                int b = Integer.parseInt(parts[1]);
                int c = Integer.parseInt(parts[2]);
                String label = parts.length >= 4 ? parts[3] : "?";
                data.add(new DataPoint(a, b, c, label));
            }
        }

        reader.close();
        return data;
    }

    // Ước lượng các xác suất có điều kiện
    public static ConditionalProbabilities estimateProbabilities(List<DataPoint> data) {
        ConditionalProbabilities probs = new ConditionalProbabilities();

        // Đếm số lượng mỗi class
        int countPlus = 0;
        int countMinus = 0;

        // Đếm số lần mỗi feature = 1 cho mỗi class khi label là '+' hoặc '-'
        int[] countFeaturePlusWhen1 = new int[3];
        int[] countFeatureMinusWhen1 = new int[3];

        for (DataPoint point : data) {
            if (point.label.equals("+")) {
                countPlus++;
                for (int i = 0; i < 3; i++) {
                    if (point.features[i] == 1) {
                        countFeaturePlusWhen1[i]++;
                    }
                }
            } else if (point.label.equals("-")) {
                countMinus++;
                for (int i = 0; i < 3; i++) {
                    if (point.features[i] == 1) {
                        countFeatureMinusWhen1[i]++;
                    }
                }
            }
        }

        // Tính xác suất tiên nghiệm
        int total = data.size();
        probs.priorPlus = (double) countPlus / total;
        probs.priorMinus = (double) countMinus / total;

        // Tính xác suất có điều kiện
        for (int i = 0; i < 3; i++) {
            probs.probGivenPlus[i] = countPlus > 0 ?
                    (double) countFeaturePlusWhen1[i] / countPlus : 0;
            probs.probGivenMinus[i] = countMinus > 0 ?
                    (double) countFeatureMinusWhen1[i] / countMinus : 0;
        }

        // Chi tiết tính toán
        System.out.println("\nChi tiết tính toán:");
        System.out.println("  Tổng số mẫu: " + total);
        System.out.println("  Số mẫu '+': " + countPlus);
        System.out.println("  Số mẫu '-': " + countMinus);

        String[] featureNames = {"A", "B", "C"};
        System.out.println("\n  Đếm feature = 1 cho class '+':");
        for (int i = 0; i < 3; i++) {
            System.out.printf("    %s = 1 trong '+': %d/%d = %.4f\n",
                    featureNames[i], countFeaturePlusWhen1[i], countPlus, probs.probGivenPlus[i]);
        }

        System.out.println("\n  Đếm feature = 1 cho class '-':");
        for (int i = 0; i < 3; i++) {
            System.out.printf("    %s = 1 trong '-': %d/%d = %.4f\n",
                    featureNames[i], countFeatureMinusWhen1[i], countMinus, probs.probGivenMinus[i]);
        }

        return probs;
    }

    // Phân lớp điểm test
    public static void classifyWithNaiveBayes(DataPoint testPoint, ConditionalProbabilities probs) {
        System.out.printf("\nĐiểm test: A=%d, B=%d, C=%d\n",
                testPoint.features[0], testPoint.features[1], testPoint.features[2]);

        // Tính P(X|+) * P(+)
        double probPlus = probs.priorPlus;
        System.out.printf("\nTính toán cho class '+':\n");
        System.out.printf("  P(+) = %.4f\n", probPlus);

        String[] featureNames = {"A", "B", "C"};
        for (int i = 0; i < 3; i++) {
            double p;
            if (testPoint.features[i] == 1) {
                p = probs.probGivenPlus[i];
                System.out.printf("  P(%s=1|+) = %.4f\n", featureNames[i], p);
            } else {
                p = 1 - probs.probGivenPlus[i];
                System.out.printf("  P(%s=0|+) = 1 - %.4f = %.4f\n",
                        featureNames[i], probs.probGivenPlus[i], p);
            }
            probPlus *= p;
        }
        System.out.printf("  P(X|+) * P(+) = %.6f\n", probPlus);

        // Tính P(X|-) * P(-)
        double probMinus = probs.priorMinus;
        System.out.printf("\nTính toán cho class '-':\n");
        System.out.printf("  P(-) = %.4f\n", probMinus);

        for (int i = 0; i < 3; i++) {
            double p;
            if (testPoint.features[i] == 1) {
                p = probs.probGivenMinus[i];
                System.out.printf("  P(%s=1|-) = %.4f\n", featureNames[i], p);
            } else {
                p = 1 - probs.probGivenMinus[i];
                System.out.printf("  P(%s=0|-) = 1 - %.4f = %.4f\n",
                        featureNames[i], probs.probGivenMinus[i], p);
            }
            probMinus *= p;
        }
        System.out.printf("  P(X|-) * P(-) = %.6f\n", probMinus);

        // So sánh và phân lớp
        System.out.println("\n" + "-".repeat(70));
        System.out.println("Kết quả:");
        String predicted = probPlus > probMinus ? "+" : "-";
        System.out.printf("  P(X|+) * P(+) = %.6f\n", probPlus);
        System.out.printf("  P(X|-) * P(-) = %.6f\n", probMinus);
        System.out.printf("\n  → Lớp được dự đoán: '%s' (%.6f > %.6f)\n",
                predicted, Math.max(probPlus, probMinus), Math.min(probPlus, probMinus));

        if (!testPoint.label.equals("?")) {
            System.out.printf("  → Lớp thực tế: '%s'\n", testPoint.label);
            System.out.printf("  → Kết quả: %s\n",
                    predicted.equals(testPoint.label) ? "ĐÚNG ✓" : "SAI ✗");
        }
    }
}