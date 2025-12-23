package org.example.ex6;

import java.io.*;
import java.util.*;

public class DecisionTree {

    static class DataPoint {
        String a1;
        String a2;
        double a3;
        String targetClass;

        public DataPoint(String a1, String a2, double a3, String targetClass) {
            this.a1 = a1;
            this.a2 = a2;
            this.a3 = a3;
            this.targetClass = targetClass;
        }

        @Override
        public String toString() {
            return String.format("a1 = %s, a2 = %s, a3 = %.1f -> %s", a1, a2, a3, targetClass);
        }
    }

    public static void main(String[] args) {
        try {
            List<DataPoint> data = readTrainingData("dataset");
            System.out.println("Cho bộ dữ liệu training có " + data.size() + " điểm training");
            System.out.println("\nTraining Dataset:");
            System.out.printf("%-10s %-5s %-5s %-8s %-15s\n", "Instance", "a1", "a2", "a3", "Target Class");
            System.out.println("-".repeat(60));
            for (int i = 0; i < data.size(); i++) {
                DataPoint p = data.get(i);
                System.out.printf("%-10d %-5s %-5s %-8.1f %-15s\n", i + 1, p.a1, p.a2, p.a3, p.targetClass);
            }

            // Tính entropy với positive class
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Tính entropy với positive class");
            System.out.println("=".repeat(80));
            double entropy = calculateEntropy(data, "+");
            System.out.printf("\nEntropy = %.4f\n", entropy);

            // Tính information gain cho a1 và a2
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Tính information gain cho a1 và a2");
            System.out.println("=".repeat(80));

            double gainA1 = calculateInformationGain(data, "a1", entropy);
            System.out.printf("\nInformation Gain(a1) = %.4f\n", gainA1);

            double gainA2 = calculateInformationGain(data, "a2", entropy);
            System.out.printf("\nInformation Gain(a2) = %.4f\n", gainA2);

            // Tính Gini index cho a1 và a2. Gini = 1 - (P(+)² + P(-)²)
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Tính Gini index cho a1 và a2");
            System.out.println("=".repeat(80));

            double giniA1 = calculateGiniSplit(data, "a1");
            System.out.printf("\nGini Index cho a1 = %.4f\n", giniA1);

            double giniA2 = calculateGiniSplit(data, "a2");
            System.out.printf("\nGini Index cho a2 = %.4f\n", giniA2);

            System.out.println("\n" + "-".repeat(80));
            System.out.println("So sánh:");
            System.out.printf("  Gini(a1) = %.4f\n", giniA1);
            System.out.printf("  Gini(a2) = %.4f\n", giniA2);
            String bestSplit = giniA1 < giniA2 ? "a1" : "a2";
            System.out.printf("\n  → Kết quả: %s (Gini thấp hơn = %.4f)\n",
                    bestSplit, Math.min(giniA1, giniA2));

        } catch (IOException e) {
            System.err.println("Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = DecisionTree.class.getClassLoader()
                .getResourceAsStream("ex6/" + filename);

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
                data.add(new DataPoint(parts[0], parts[1], Double.parseDouble(parts[2]), parts[3]));
            }
        }

        reader.close();
        return data;
    }

    // Tính Entropy = -P(+) × log₂(P(+)) - P(-) × log₂(P(-))
    public static double calculateEntropy(List<DataPoint> data, String positiveClass) {
        int total = data.size();
        int positiveCount = 0;

        for (DataPoint point : data) {
            if (point.targetClass.equals(positiveClass)) {
                positiveCount++;
            }
        }

        int negativeCount = total - positiveCount;

        System.out.println("\nTính Entropy:");
        System.out.println("  Tổng số mẫu: " + total);
        System.out.println("  Positive (+) instances: " + positiveCount);
        System.out.println("  Negative (-) instances: " + negativeCount);

        double pPositive = (double) positiveCount / total;
        double pNegative = (double) negativeCount / total;

        System.out.printf("  P(+) = %d/%d = %.4f\n", positiveCount, total, pPositive);
        System.out.printf("  P(-) = %d/%d = %.4f\n", negativeCount, total, pNegative);

        double entropy = 0.0;
        if (pPositive > 0) {
            entropy -= pPositive * (Math.log(pPositive) / Math.log(2));
        }
        if (pNegative > 0) {
            entropy -= pNegative * (Math.log(pNegative) / Math.log(2));
        }

        System.out.printf("\n  Entropy = -%.4f × log2(%.4f) - %.4f × log2(%.4f)\n",
                pPositive, pPositive, pNegative, pNegative);

        return entropy;
    }

    // Tính Information Gain = Entropy(parent) - Weighted Entropy
    public static double calculateInformationGain(List<DataPoint> data, String attribute, double parentEntropy) {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Tính Information Gain cho thuộc tính: " + attribute);
        System.out.println("-".repeat(80));

        Map<String, List<DataPoint>> partitions = new HashMap<>();

        // Phân chia data theo attribute
        for (DataPoint point : data) {
            String value = attribute.equals("a1") ? point.a1 : point.a2;
            partitions.computeIfAbsent(value, k -> new ArrayList<>()).add(point);
        }

        double weightedEntropy = 0.0;
        int total = data.size();

        System.out.println("\nPartitions:");
        for (Map.Entry<String, List<DataPoint>> entry : partitions.entrySet()) {
            String value = entry.getKey();
            List<DataPoint> subset = entry.getValue();
            int subsetSize = subset.size();

            // Đếm positive và negative trong subset
            int positiveCount = 0;
            for (DataPoint point : subset) {
                if (point.targetClass.equals("+")) {
                    positiveCount++;
                }
            }
            int negativeCount = subsetSize - positiveCount;

            System.out.printf("\n  %s = %s: %d instances (%d positive, %d negative)\n",
                    attribute, value, subsetSize, positiveCount, negativeCount);

            double pPositive = (double) positiveCount / subsetSize;
            double pNegative = (double) negativeCount / subsetSize;

            double entropy = 0.0;
            if (pPositive > 0) {
                entropy -= pPositive * (Math.log(pPositive) / Math.log(2));
            }
            if (pNegative > 0) {
                entropy -= pNegative * (Math.log(pNegative) / Math.log(2));
            }

            System.out.printf("    Entropy = %.4f\n", entropy);

            double weight = (double) subsetSize / total;
            weightedEntropy += weight * entropy;

            System.out.printf("    Weight = %d/%d = %.4f\n", subsetSize, total, weight);
        }

        double informationGain = parentEntropy - weightedEntropy;

        System.out.println("\n  Weighted Entropy = " + String.format("%.4f", weightedEntropy));
        System.out.printf("  Information Gain = %.4f - %.4f = %.4f\n",
                parentEntropy, weightedEntropy, informationGain);

        return informationGain;
    }

    // Tính Gini Index cho split
    public static double calculateGiniSplit(List<DataPoint> data, String attribute) {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Tính Gini Index cho: " + attribute);
        System.out.println("-".repeat(80));

        Map<String, List<DataPoint>> partitions = new HashMap<>();

        // Phân chia data theo attribute
        for (DataPoint point : data) {
            String value = attribute.equals("a1") ? point.a1 : point.a2;
            partitions.computeIfAbsent(value, k -> new ArrayList<>()).add(point);
        }

        double weightedGini = 0.0;
        int total = data.size();

        System.out.println("\nPartitions:");
        for (Map.Entry<String, List<DataPoint>> entry : partitions.entrySet()) {
            String value = entry.getKey();
            List<DataPoint> subset = entry.getValue();
            int subsetSize = subset.size();

            // Đếm positive và negative trong subset
            int positiveCount = 0;
            for (DataPoint point : subset) {
                if (point.targetClass.equals("+")) {
                    positiveCount++;
                }
            }
            int negativeCount = subsetSize - positiveCount;

            System.out.printf("\n  %s = %s: %d mẫu (%d positive, %d negative)\n",
                    attribute, value, subsetSize, positiveCount, negativeCount);

            double pPositive = (double) positiveCount / subsetSize;
            double pNegative = (double) negativeCount / subsetSize;

            double gini = 1.0 - (pPositive * pPositive + pNegative * pNegative);

            System.out.printf("    P(+) = %.4f, P(-) = %.4f\n", pPositive, pNegative);
            System.out.printf("    Gini = 1 - (%.4f² + %.4f²) = %.4f\n",
                    pPositive, pNegative, gini);

            double weight = (double) subsetSize / total;
            weightedGini += weight * gini;

            System.out.printf("    Weight = %d/%d = %.4f\n", subsetSize, total, weight);
        }

        System.out.printf("\n  Weighted Gini Index = %.4f\n", weightedGini);

        return weightedGini;
    }
}