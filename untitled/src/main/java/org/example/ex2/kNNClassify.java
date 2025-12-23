package org.example.ex2;

import java.io.*;
import java.util.*;

public class kNNClassify {
    // Biểu diễn một mẫu
    static class DataPoint {
        double x1;
        double x2;
        String label;
        double distance;

        public DataPoint(double x1, double x2, String label) {
            this.x1 = x1;
            this.x2 = x2;
            this.label = label;
        }

        public DataPoint(DataPoint other) {
            this.x1 = other.x1;
            this.x2 = other.x2;
            this.label = other.label;
            this.distance = other.distance;
        }

        public double euclideanDistance(double qx1, double qx2) {
            return Math.sqrt(Math.pow(this.x1 - qx1, 2) + Math.pow(this.x2 - qx2, 2));
        }

        @Override
        public String toString() {
            return String.format("(%.0f,%.0f,%s,dist = %.2f)", x1, x2, label, distance);
        }
    }

    static class ConfusionMatrix {
        int tp = 0; // True Positive
        int tn = 0; // True Negative
        int fp = 0; // False Positive
        int fn = 0; // False Negative

        public void add(String actual, String predicted) {
            if (actual.equals("+") && predicted.equals("+")) tp++;
            else if (actual.equals("-") && predicted.equals("-")) tn++;
            else if (actual.equals("-") && predicted.equals("+")) fp++;
            else if (actual.equals("+") && predicted.equals("-")) fn++;
        }

        public double accuracy() {
            int total = tp + tn + fp + fn;
            return total > 0 ? (double)(tp + tn) / total : 0;
        }

        public double precision() {
            return (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
        }

        public double recall() {
            return (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
        }

        public double f1Score() {
            double p = precision();
            double r = recall();
            return (p + r) > 0 ? 2 * p * r / (p + r) : 0;
        }

        public void print() {
            System.out.println("\n  Confusion Matrix:");
            System.out.println("                Dự đoán");
            System.out.println("                +       -");
            System.out.println("  Thực tế +     " + tp + "       " + fn);
            System.out.println("          -     " + fp + "       " + tn);
            System.out.printf("\n  Accuracy:  %.2f%%\n", accuracy() * 100);
            System.out.printf("  Precision: %.2f%%\n", precision() * 100);
            System.out.printf("  Recall:    %.2f%%\n", recall() * 100);
            System.out.printf("  F1-Score:  %.2f%%\n", f1Score() * 100);
        }
    }

    public static void main(String[] args) {
        try {
            // Đọc toàn bộ dataset
            List<DataPoint> allData = readDataset("ex6/dataset");
            System.out.println("Cho bộ dữ liệu training có " + allData.size() + " điểm dữ liệu");
            System.out.println("Training dataset:");
            for (DataPoint p : allData) {
                System.out.printf("  x = %.1f, y = %.1f, z = %s\n", p.x1, p.x2, p.label);
            }

            // Phân lớp điểm test (7,5)
            System.out.println("\n" + "=".repeat(70));
            System.out.println("Phân loại điểm test (7,5)");
            System.out.println("=".repeat(70));

            classifyQueryPoint(allData, 7, 5);

            // Chia dataset và đánh giá
            System.out.println("\n\n" + "=".repeat(70));
            System.out.println("Đánh giá mô hình");
            System.out.println("=".repeat(70));

            evaluateModel(allData);

        } catch (IOException e) {
            System.err.println("Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = kNNClassify.class.getClassLoader()
                .getResourceAsStream("ex2/" + filename);

        if (is != null) {
            return new BufferedReader(new InputStreamReader(is));
        }
        throw new FileNotFoundException("Không tìm thấy file: " + filename);
    }

    public static List<DataPoint> readDataset(String filename) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            String[] parts = line.split("\\s+");
            if (parts.length >= 3) {
                double x1 = Double.parseDouble(parts[0]);
                double x2 = Double.parseDouble(parts[1]);
                String label = parts[2];
                data.add(new DataPoint(x1, x2, label));
            }
        }

        reader.close();
        return data;
    }

    // Phân lớp điểm test (7,5)
    public static void classifyQueryPoint(List<DataPoint> trainingData, double qx1, double qx2) {
                // Tính khoảng cách
        List<DataPoint> dataset = new ArrayList<>();
        for (DataPoint p : trainingData) {
            DataPoint copy = new DataPoint(p);
            copy.distance = p.euclideanDistance(qx1, qx2);
            dataset.add(copy);
        }

        dataset.sort(Comparator.comparingDouble(p -> p.distance));

        System.out.println("\nCác điểm được sắp xếp theo khoảng cách đến điểm test:");
        for (int i = 0; i < Math.min(10, dataset.size()); i++) {
            System.out.printf("%2d. %s\n", i + 1, dataset.get(i));
        }

        int[] kValues = {1, 3, 5, 9};
        System.out.println("\n" + "=".repeat(70));
        System.out.println("Kết quả: ");
        System.out.println("=".repeat(70));

        for (int k : kValues) {
            String result = classifyKNN(dataset, k);
            System.out.println(result);
        }
    }

    // Đánh giá mô hình
    public static void evaluateModel(List<DataPoint> allData) {
        //Random dữ liệu đảm bảo các mẫu + và - được phân bố đều giữa train và test.
        List<DataPoint> shuffledData = new ArrayList<>(allData);
        Collections.shuffle(shuffledData, new Random(42));

        // Chia 80-20
        int trainSize = (int)(shuffledData.size() * 0.8);
        List<DataPoint> trainData = shuffledData.subList(0, trainSize);
        List<DataPoint> testData = shuffledData.subList(trainSize, shuffledData.size());

        System.out.println("\nTrain size: " + trainData.size());
        System.out.println("Test size: " + testData.size());

        int[] kValues = {1, 3, 5, 7, 9};

        System.out.println("\n" + "=".repeat(70));
        System.out.println("Đánh giá mô hình với các giá trị k");
        System.out.println("=".repeat(70));

        for (int k : kValues) {
            System.out.println("\n" + "-".repeat(70));
            System.out.println("K = " + k);
            System.out.println("-".repeat(70));

            ConfusionMatrix cm = new ConfusionMatrix();
            int correct = 0;

            for (DataPoint testPoint : testData) {
                // Tính khoảng cách đến tất cả điểm training
                List<DataPoint> distances = new ArrayList<>();
                for (DataPoint trainPoint : trainData) {
                    DataPoint copy = new DataPoint(trainPoint);
                    copy.distance = trainPoint.euclideanDistance(testPoint.x1, testPoint.x2);
                    distances.add(copy);
                }

                // Sắp xếp, lấy k láng giềng
                distances.sort(Comparator.comparingDouble(p -> p.distance));

                String predicted = majorityVote(distances, k);
                cm.add(testPoint.label, predicted);

                if (predicted.equals(testPoint.label)) {
                    correct++;
                }
            }

            cm.print();
        }

        // Kết quả
        System.out.println("\n" + "=".repeat(70));
        System.out.println("kết quả");
        System.out.println("=".repeat(70));
        System.out.printf("%-10s %-15s %-15s %-15s %-15s\n",
                "k", "Accuracy", "Precision", "Recall", "F1-Score");
        System.out.println("-".repeat(70));

        for (int k : kValues) {
            // Tạo ma trận mới cho mỗi giá trị k
            ConfusionMatrix cm = new ConfusionMatrix();
            for (DataPoint testPoint : testData) {
                List<DataPoint> distances = new ArrayList<>();
                for (DataPoint trainPoint : trainData) {
                    DataPoint copy = new DataPoint(trainPoint);
                    copy.distance = trainPoint.euclideanDistance(testPoint.x1, testPoint.x2);
                    distances.add(copy);
                }
                distances.sort(Comparator.comparingDouble(p -> p.distance));
                String predicted = majorityVote(distances, k);
                cm.add(testPoint.label, predicted);
            }
            System.out.printf("%-10d %-15.2f %-15.2f %-15.2f %-15.2f\n",
                    k, cm.accuracy()*100, cm.precision()*100,
                    cm.recall()*100, cm.f1Score()*100);
        }
    }

    // Dự đoán nhãn cho điểm test
    public static String classifyKNN(List<DataPoint> sortedData, int k) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("\nVới k = %d:\n", k));
        sb.append(String.format("  %d láng giêng gần nhất: ", k));

        Map<String, Integer> votes = new HashMap<>();
        for (int i = 0; i < k && i < sortedData.size(); i++) {
            DataPoint point = sortedData.get(i);
            votes.put(point.label, votes.getOrDefault(point.label, 0) + 1);
            sb.append(String.format("(%.0f,%.0f,%s) ", point.x1, point.x2, point.label));
        }

        int plusVotes = votes.getOrDefault("+", 0);
        int minusVotes = votes.getOrDefault("-", 0);

        sb.append(String.format("\n  Phiếu bầu: '+' = %d, '-' = %d", plusVotes, minusVotes));

        String prediction = plusVotes > minusVotes ? "+" : "-";
        sb.append(String.format("\n  → Kết quả: điểm được phân loại là '%s'", prediction));

        return sb.toString();
    }

    // Dự đoán nhãn cho ma trận
    public static String majorityVote(List<DataPoint> sortedData, int k) {
        Map<String, Integer> votes = new HashMap<>();
        for (int i = 0; i < k && i < sortedData.size(); i++) {
            DataPoint point = sortedData.get(i);
            votes.put(point.label, votes.getOrDefault(point.label, 0) + 1);
        }

        int plusVotes = votes.getOrDefault("+", 0);
        int minusVotes = votes.getOrDefault("-", 0);

        return plusVotes > minusVotes ? "+" : "-";
    }
}