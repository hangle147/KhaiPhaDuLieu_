package org.example.ex7;

import java.io.*;
import java.util.*;

public class DecisionTree {

    static class Customer {
        int id;
        String gender;
        String carType;
        String shirtSize;
        String classLabel;

        public Customer(int id, String gender, String carType, String shirtSize, String classLabel) {
            this.id = id;
            this.gender = gender;
            this.carType = carType;
            this.shirtSize = shirtSize;
            this.classLabel = classLabel;
        }

        @Override
        public String toString() {
            return String.format("ID=%d, Gender=%s, CarType=%s, ShirtSize=%s -> %s",
                    id, gender, carType, shirtSize, classLabel);
        }
    }

    public static void main(String[] args) {
        try {
            List<Customer> data = readTrainingData("dataset");
            System.out.println("Cho bộ dữ liệu training có " + data.size() + " mẫu");
            System.out.println("\nTraining Dataset:");
            System.out.printf("%-12s %-10s %-12s %-15s %-10s\n",
                    "Customer ID", "Gender", "Car Type", "Shirt Size", "Class");
            System.out.println("-".repeat(70));
            for (Customer c : data) {
                System.out.printf("%-12d %-10s %-12s %-15s %-10s\n",
                        c.id, c.gender, c.carType, c.shirtSize, c.classLabel);
            }

            // Gini index ở root (tất cả examples)
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Gini index ở root (tất cả examples)");
            System.out.println("=".repeat(80));
            double rootGini = calculateGiniImpurity(data);
            System.out.printf("\nGini index ở root = %.4f\n", rootGini);

            // Gini index cho Customer ID attribute
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Gini index cho Customer ID attribute");
            System.out.println("=".repeat(80));
            double giniCustomerId = calculateGiniSplit(data, "CustomerId");
            // Mã khách hàng là định danh duy nhất và không phù hợp để phân tách.
            // Mỗi lần phân tách sẽ tạo ra các nút lá với mỗi nút chỉ có 1 bản ghi.
            // Dẫn tới Gini = 0, nhưng gây ra hiện tượng overfitting.

            // Gini index cho Gender attribute
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Gini index cho Gender attribute");
            System.out.println("=".repeat(80));
            double giniGender = calculateGiniSplit(data, "Gender");

            // Gini index cho Car Type attribute
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Gini index cho Car Type attribute");
            System.out.println("=".repeat(80));
            double giniCarType = calculateGiniSplit(data, "CarType");

            // Gini index cho Shirt Size attribute
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Gini index cho Shirt Size attribute");
            System.out.println("=".repeat(80));
            double giniShirtSize = calculateGiniSplit(data, "ShirtSize");

            // So sánh và chọn attribute tốt nhất
            System.out.println("\n" + "=".repeat(80));
            System.out.println("So sánh và chọn attribute tốt nhất");
            System.out.println("=".repeat(80));

            Map<String, Double> results = new LinkedHashMap<>();
            results.put("Gender", giniGender);
            results.put("Car Type", giniCarType);
            results.put("Shirt Size", giniShirtSize);

            System.out.println("\nGini Index:");
            System.out.printf("%-20s %-15s\n", "Attribute", "Gini Index");
            System.out.println("-".repeat(40));

            double minGini = Double.MAX_VALUE;
            String bestAttribute = "";

            for (Map.Entry<String, Double> entry : results.entrySet()) {
                System.out.printf("%-20s %-15.4f\n", entry.getKey(), entry.getValue());
                if (entry.getValue() < minGini) {
                    minGini = entry.getValue();
                    bestAttribute = entry.getKey();
                }
            }

            System.out.println("\n" + "-".repeat(40));
            System.out.printf("Thuộc tình tốt nhất để phân tách: %s (Gini = %.4f)\n",
                    bestAttribute, minGini);
            // Chỉ số Gini thấp hơn cho thấy chất lượng phân tách tốt hơn.
            // Thuộc tính có chỉ số Gini thấp nhất cung cấp các phân vùng đồng nhất nhất và nên được chọn để phân tách.

        } catch (IOException e) {
            System.err.println("Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = DecisionTree.class.getClassLoader()
                .getResourceAsStream("ex7/" + filename);

        if (is != null) {
            return new BufferedReader(new InputStreamReader(is));
        }
        throw new FileNotFoundException("Không tìm thấy file: " + filename);
    }

    public static List<Customer> readTrainingData(String filename) throws IOException {
        List<Customer> data = new ArrayList<>();
        BufferedReader reader = getReader(filename);
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            String[] parts = line.split("\\s+");
            if (parts.length >= 5) {
                int id = Integer.parseInt(parts[0]);
                data.add(new Customer(id, parts[1], parts[2], parts[3], parts[4]));
            }
        }

        reader.close();
        return data;
    }

    // Tính Gini Impurity cho một tập dữ liệu
    public static double calculateGiniImpurity(List<Customer> data) {
        if (data.isEmpty()) return 0.0;

        Map<String, Integer> classCount = new HashMap<>();
        for (Customer c : data) {
            classCount.put(c.classLabel, classCount.getOrDefault(c.classLabel, 0) + 1);
        }

        int total = data.size();
        System.out.println("\nTính Gini Impurity:");
        System.out.println("  Total instances: " + total);
        System.out.println("  Class distribution:");

        double gini = 1.0;
        for (Map.Entry<String, Integer> entry : classCount.entrySet()) {
            int count = entry.getValue();
            double probability = (double) count / total;
            System.out.printf("    Class %s: %d mẫu (P = %.4f)\n",
                    entry.getKey(), count, probability);
            gini -= probability * probability;
        }

        System.out.print("\n  Gini = 1");
        for (Map.Entry<String, Integer> entry : classCount.entrySet()) {
            double probability = (double) entry.getValue() / total;
            System.out.printf(" - (%.4f)²", probability);
        }
        System.out.printf(" = %.4f\n", gini);

        return gini;
    }

    // Tính Gini Index cho split theo attribute
    public static double calculateGiniSplit(List<Customer> data, String attribute) {
        Map<String, List<Customer>> partitions = new HashMap<>();

        // Phân chia data theo attribute
        for (Customer c : data) {
            String value = getAttributeValue(c, attribute);
            partitions.computeIfAbsent(value, k -> new ArrayList<>()).add(c);
        }

        double weightedGini = 0.0;
        int total = data.size();

        // Phân hoạch
        System.out.println("\nPartitions:");
        for (Map.Entry<String, List<Customer>> entry : partitions.entrySet()) {
            String value = entry.getKey();
            List<Customer> subset = entry.getValue();
            int subsetSize = subset.size();

            System.out.printf("\n  %s = %s: %d mẫu\n", attribute, value, subsetSize);

            // Đếm class distribution trong subset
            Map<String, Integer> classCount = new HashMap<>();
            for (Customer c : subset) {
                classCount.put(c.classLabel, classCount.getOrDefault(c.classLabel, 0) + 1);
            }

            System.out.println("    Class distribution:");
            double gini = 1.0;
            for (Map.Entry<String, Integer> classEntry : classCount.entrySet()) {
                int count = classEntry.getValue();
                double probability = (double) count / subsetSize;
                System.out.printf("      Class %s: %d (P = %.4f)\n",
                        classEntry.getKey(), count, probability);
                gini -= probability * probability;
            }

            System.out.printf("    Gini = %.4f\n", gini);

            double weight = (double) subsetSize / total;
            weightedGini += weight * gini;

            System.out.printf("    Weight = %d/%d = %.4f\n", subsetSize, total, weight);
            System.out.printf("    Contribution = %.4f × %.4f = %.4f\n",
                    weight, gini, weight * gini);
        }

        System.out.printf("\n  Weighted Gini Index = %.4f\n", weightedGini);

        return weightedGini;
    }

    private static String getAttributeValue(Customer c, String attribute) {
        switch (attribute) {
            case "Gender": return c.gender;
            case "CarType": return c.carType;
            case "ShirtSize": return c.shirtSize;
            default: return "";
        }
    }
}