package org.example.ex8;

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
            // Đọc training data
            List<DataPoint> data = readTrainingData("dataset");
            System.out.println("Cho bộ dữ liệu training có " + data.size() + " điểm training");
            System.out.println("\nTraining Dataset:");
            System.out.printf("%-10s %-5s %-5s %-8s %-15s\n", "Instance", "a1", "a2", "a3", "Target Class");
            System.out.println("-".repeat(60));
            for (int i = 0; i < data.size(); i++) {
                DataPoint p = data.get(i);
                System.out.printf("%-10d %-5s %-5s %-8.1f %-15s\n", i + 1, p.a1, p.a2, p.a3, p.targetClass);
            }

            // Entropy với positive class
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Entropy với positive class'+'");
            System.out.println("=".repeat(80));
            double entropy = calculateEntropy(data, "+");
            System.out.printf("\nEntropy = %.4f\n", entropy);

            // Phân chia tốt nhất theo Information Gain
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Phân chia tốt nhất theo Information Gain");
            System.out.println("=".repeat(80));

            double gainA1 = calculateInformationGain(data, "a1", entropy);
            double gainA2 = calculateInformationGain(data, "a2", entropy);
            double gainA3 = calculateInformationGainContinuous(data, "a3", entropy);

            System.out.println("\n" + "-".repeat(80));
            System.out.println("Information Gain Summary:");
            System.out.println("-".repeat(80));
            System.out.printf("  Gain(a1) = %.4f\n", gainA1);
            System.out.printf("  Gain(a2) = %.4f\n", gainA2);
            System.out.printf("  Gain(a3) = %.4f\n", gainA3);

            double maxGain = Math.max(gainA1, Math.max(gainA2, gainA3));
            String bestAttr = "";
            if (maxGain == gainA1) bestAttr = "a1";
            else if (maxGain == gainA2) bestAttr = "a2";
            else bestAttr = "a3";

            System.out.printf("\n  → Phân chia tốt nhất: %s (Information Gain = %.4f)\n", bestAttr, maxGain);

            // Phân chia tốt nhất theo chỉ số Gini
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Phân chia tốt nhất theo chỉ số Gini");
            System.out.println("=".repeat(80));

            double giniA1 = calculateGiniSplit(data, "a1");
            double giniA2 = calculateGiniSplit(data, "a2");
            double giniA3 = calculateGiniSplitContinuous(data, "a3");

            System.out.println("\n" + "-".repeat(80));
            System.out.println("Chỉ số Gini Summary:");
            System.out.println("-".repeat(80));
            System.out.printf("  Gini(a1) = %.4f\n", giniA1);
            System.out.printf("  Gini(a2) = %.4f\n", giniA2);
            System.out.printf("  Gini(a3) = %.4f\n", giniA3);

            double minGini = Math.min(giniA1, Math.min(giniA2, giniA3));
            String bestAttrGini = "";
            if (minGini == giniA1) bestAttrGini = "a1";
            else if (minGini == giniA2) bestAttrGini = "a2";
            else bestAttrGini = "a3";

            System.out.printf("\n  → Phân chia tốt nhất: %s (Chỉ số Gini = %.4f)\n", bestAttrGini, minGini);

        } catch (IOException e) {
            System.err.println("Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = DecisionTree.class.getClassLoader()
                .getResourceAsStream("ex8/" + filename);

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

    public static double calculateEntropy(List<DataPoint> data, String positiveClass) {
        int total = data.size();
        int positiveCount = 0;

        for (DataPoint point : data) {
            if (point.targetClass.equals(positiveClass)) {
                positiveCount++;
            }
        }

        int negativeCount = total - positiveCount;

        System.out.println("\nCalculating Entropy:");
        System.out.println("  Total instances: " + total);
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

    //Tính Information Gain
    public static double calculateInformationGain(List<DataPoint> data, String attribute, double parentEntropy) {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Information Gain cho thuộc tính: " + attribute);
        System.out.println("-".repeat(80));

        Map<String, List<DataPoint>> partitions = new HashMap<>();

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

            int positiveCount = 0;
            for (DataPoint point : subset) {
                if (point.targetClass.equals("+")) {
                    positiveCount++;
                }
            }
            int negativeCount = subsetSize - positiveCount;

            System.out.printf("\n  %s = %s: %d Mẫu (%d positive, %d negative)\n",
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

        System.out.printf("\n  Weighted Entropy = %.4f\n", weightedEntropy);
        System.out.printf("  Information Gain = %.4f - %.4f = %.4f\n",
                parentEntropy, weightedEntropy, informationGain);

        return informationGain;
    }

    public static double calculateInformationGainContinuous(List<DataPoint> data, String attribute, double parentEntropy) {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Tính Information Gain cho thuộc tính liên tục: " + attribute);
        System.out.println("-".repeat(80));

        // Sắp xếp data theo a3
        List<DataPoint> sortedData = new ArrayList<>(data);
        sortedData.sort(Comparator.comparingDouble(p -> p.a3));

        System.out.println("\nSắp xếp theo a3:");
        for (DataPoint p : sortedData) {
            System.out.printf("  a3=%.1f -> %s\n", p.a3, p.targetClass);
        }

        // Tìm các split points có thể (giữa các giá trị khác nhau)
        Set<Double> splitPoints = new TreeSet<>();
        for (int i = 0; i < sortedData.size() - 1; i++) {
            if (sortedData.get(i).a3 != sortedData.get(i + 1).a3) {
                double splitPoint = (sortedData.get(i).a3 + sortedData.get(i + 1).a3) / 2;
                splitPoints.add(splitPoint);
            }
        }

        System.out.println("\nĐiểm có thể tách: " + splitPoints);

        double maxGain = -1;
        double bestSplit = 0;

        // Thử từng split point
        for (double splitPoint : splitPoints) {
            List<DataPoint> left = new ArrayList<>();
            List<DataPoint> right = new ArrayList<>();

            for (DataPoint p : data) {
                if (p.a3 <= splitPoint) {
                    left.add(p);
                } else {
                    right.add(p);
                }
            }

            System.out.printf("\nPhân tách a3 <= %.1f:\n", splitPoint);
            System.out.printf("  Trái: %d mẫu\n", left.size());
            System.out.printf("  Phải: %d mẫu\n", right.size());

            double leftEntropy = calculateEntropyForSubset(left);
            double rightEntropy = calculateEntropyForSubset(right);

            double weightedEntropy = ((double) left.size() / data.size()) * leftEntropy +
                    ((double) right.size() / data.size()) * rightEntropy;

            double gain = parentEntropy - weightedEntropy;

            System.out.printf("  Weighted Entropy = %.4f\n", weightedEntropy);
            System.out.printf("  Information Gain = %.4f\n", gain);

            if (gain > maxGain) {
                maxGain = gain;
                bestSplit = splitPoint;
            }
        }

        System.out.printf("\n  → Điểm phân tách tốt nhất: a3 <= %.1f\n", bestSplit);
        System.out.printf("  → Information Gain lớn nhất = %.4f\n", maxGain);

        return maxGain;
    }

    //Tính chỉ số Gini cho thuộc tính
    public static double calculateGiniSplit(List<DataPoint> data, String attribute) {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Tính chỉ số Gini cho thuộc tính: " + attribute);
        System.out.println("-".repeat(80));

        Map<String, List<DataPoint>> partitions = new HashMap<>();

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

            System.out.printf("    Gini = 1 - (%.4f² + %.4f²) = %.4f\n",
                    pPositive, pNegative, gini);

            double weight = (double) subsetSize / total;
            weightedGini += weight * gini;

            System.out.printf("    Weight = %d/%d = %.4f\n", subsetSize, total, weight);
        }

        System.out.printf("\n Chỉ số Weighted Gini = %.4f\n", weightedGini);

        return weightedGini;
    }

    // Tính chỉ số Gini cho thuộc tính liên tục
    public static double calculateGiniSplitContinuous(List<DataPoint> data, String attribute) {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Tính chỉ số Gini cho thuộc tính liên tục: " + attribute);
        System.out.println("-".repeat(80));

        List<DataPoint> sortedData = new ArrayList<>(data);
        sortedData.sort(Comparator.comparingDouble(p -> p.a3));

        Set<Double> splitPoints = new TreeSet<>();
        for (int i = 0; i < sortedData.size() - 1; i++) {
            if (sortedData.get(i).a3 != sortedData.get(i + 1).a3) {
                double splitPoint = (sortedData.get(i).a3 + sortedData.get(i + 1).a3) / 2;
                splitPoints.add(splitPoint);
            }
        }

        double minGini = Double.MAX_VALUE;
        double bestSplit = 0;

        for (double splitPoint : splitPoints) {
            List<DataPoint> left = new ArrayList<>();
            List<DataPoint> right = new ArrayList<>();

            for (DataPoint p : data) {
                if (p.a3 <= splitPoint) {
                    left.add(p);
                } else {
                    right.add(p);
                }
            }

            double leftGini = calculateGiniForSubset(left);
            double rightGini = calculateGiniForSubset(right);

            double weightedGini = ((double) left.size() / data.size()) * leftGini +
                    ((double) right.size() / data.size()) * rightGini;

            System.out.printf("\nSplit at a3 <= %.1f: Gini = %.4f\n", splitPoint, weightedGini);

            if (weightedGini < minGini) {
                minGini = weightedGini;
                bestSplit = splitPoint;
            }
        }

        System.out.printf("\n  → Điểm phân tách tốt nhất: a3 <= %.1f\n", bestSplit);
        System.out.printf("  → Chỉ số Gini nhỏ nhất = %.4f\n", minGini);

        return minGini;
    }

    // Tính entropy cho tập con
    private static double calculateEntropyForSubset(List<DataPoint> subset) {
        if (subset.isEmpty()) return 0.0;

        int positiveCount = 0;
        for (DataPoint p : subset) {
            if (p.targetClass.equals("+")) positiveCount++;
        }

        int negativeCount = subset.size() - positiveCount;
        double pPos = (double) positiveCount / subset.size();
        double pNeg = (double) negativeCount / subset.size();

        double entropy = 0.0;
        if (pPos > 0) entropy -= pPos * (Math.log(pPos) / Math.log(2));
        if (pNeg > 0) entropy -= pNeg * (Math.log(pNeg) / Math.log(2));

        return entropy;
    }

    // Tính gini cho tập con
    private static double calculateGiniForSubset(List<DataPoint> subset) {
        if (subset.isEmpty()) return 0.0;

        int positiveCount = 0;
        for (DataPoint p : subset) {
            if (p.targetClass.equals("+")) positiveCount++;
        }

        int negativeCount = subset.size() - positiveCount;
        double pPos = (double) positiveCount / subset.size();
        double pNeg = (double) negativeCount / subset.size();

        return 1.0 - (pPos * pPos + pNeg * pNeg);
    }
}
