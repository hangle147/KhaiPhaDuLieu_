package org.example.ex1;

import java.io.*;
import java.util.*;

public class kNNClassify {
    // Biểu diễn một mẫu
    static class DataPoint {
        double x;
        String y;
        double distance;

        public DataPoint(double x, String y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return String.format("(x = %.1f, y = %s, dist = %.2f)", x, y, distance);
        }
    }

    public static void main(String[] args) {
        try {
            List<DataPoint> trainingData = readTrainingData("ex3/training");
            System.out.println("Cho bộ dữ liệu training có " + trainingData.size() + " điểm training");
            System.out.println("Training dataset:");
            for (DataPoint p : trainingData) {
                System.out.printf("  x = %.1f, y = %s\n", p.x, p.y);
            }

            List<Double> testingData = readTestingData("testing");
            System.out.println("\nCho bộ dữ liệu testing có " + testingData.size() + " điểm testing");
            System.out.println("Testing dataset: " + testingData);

            List<Integer> kValues = readKValues("k_value");
            System.out.println("\nCác giá trị k: " + kValues);

            // Phân lớp
            for (double z : testingData) {
                System.out.println("\n" + "=".repeat(50));
                System.out.println("PHÂN LOẠI ĐIỂM z = " + z);
                System.out.println("=".repeat(50));

                // Tính khoảng cách
                List<DataPoint> dataset = new ArrayList<>();
                for (DataPoint p : trainingData) {
                    DataPoint copy = new DataPoint(p.x, p.y);
                    copy.distance = Math.abs(copy.x - z);
                    dataset.add(copy);
                }

                // Sắp xếp
                dataset.sort(Comparator.comparingDouble(p -> p.distance));

                System.out.println("\nCác điểm được sắp xếp theo khoảng cách:");
                for (int i = 0; i < dataset.size(); i++) {
                    System.out.printf("%2d. %s%n", i + 1, dataset.get(i));
                }

                System.out.println("\n" + "=".repeat(50));
                System.out.println("KẾT QUẢ PHÂN LOẠI:");
                System.out.println("=".repeat(50));

                for (int k : kValues) {
                    System.out.println(classifyKNN(dataset, k));
                }
            }

        } catch (IOException e) {
            System.err.println("Lỗi đọc dữ liệu: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = kNNClassify.class.getClassLoader().getResourceAsStream("ex1/" + filename);

        if (is != null) {
            return new BufferedReader(new InputStreamReader(is));
        }

        throw new FileNotFoundException(
                "Không tìm thấy file: " + filename + "\n" +
                        "Vui lòng tạo file hoặc đặt đúng vị trí!"
        );
    }

    public static List<DataPoint> readTrainingData(String filename) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            String[] parts = line.split("\\s+");
            if (parts.length >= 2) {
                double x = Double.parseDouble(parts[0]);
                String y = parts[1];
                data.add(new DataPoint(x, y));
            }
        }

        reader.close();
        return data;
    }

    public static List<Double> readTestingData(String filename) throws IOException {
        List<Double> data = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            data.add(Double.parseDouble(line));
        }

        reader.close();
        return data;
    }

    public static List<Integer> readKValues(String filename) throws IOException {
        List<Integer> kValues = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            String[] parts = line.split("\\s+");
            for (String part : parts) {
                kValues.add(Integer.parseInt(part));
            }
        }

        reader.close();
        return kValues;
    }

    public static String classifyKNN(List<DataPoint> sortedData, int k) {
        Map<String, Integer> votes = new HashMap<>();

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("\nVới k = %d:\n", k));
        sb.append(String.format("  %d hàng xóm gần nhất: ", k));

        // Lấy k điểm gần nhất
        for (int i = 0; i < k && i < sortedData.size(); i++) {
            DataPoint point = sortedData.get(i);
            votes.put(point.y, votes.getOrDefault(point.y, 0) + 1);
            sb.append(String.format("(%.1f, %s) ", point.x, point.y));
        }

        // Đếm số phiếu
        int plusVotes = votes.getOrDefault("+", 0);
        int minusVotes = votes.getOrDefault("-", 0);

        sb.append(String.format("\n  Phiếu bầu: '+' = %d, '-' = %d", plusVotes, minusVotes));

        // Phân loại theo số phiếu
        String prediction = plusVotes > minusVotes ? "+" : "-";
        sb.append(String.format("\n  → z được phân loại là '%s'", prediction));

        return sb.toString();
    }
}