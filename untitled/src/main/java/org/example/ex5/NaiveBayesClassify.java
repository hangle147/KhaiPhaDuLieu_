package org.example.ex5;

import java.io.*;
import java.util.*;

public class NaiveBayesClassify {

    static class DataPoint {
        String outlook;
        String temp;
        String humidity;
        String windy;
        String play;

        public DataPoint(String outlook, String temp, String humidity, String windy, String play) {
            this.outlook = outlook;
            this.temp = temp;
            this.humidity = humidity;
            this.windy = windy;
            this.play = play;
        }

        @Override
        public String toString() {
            return String.format("Outlook=%s, Temp=%s, Humidity=%s, Windy=%s -> %s",
                    outlook, temp, humidity, windy, play);
        }
    }

    //Bảng xác suất
    static class ProbabilityTable {
        Map<String, Map<String, Double>> probYes = new HashMap<>();
        Map<String, Map<String, Double>> probNo = new HashMap<>();
        double priorYes;
        double priorNo;

        public void print() {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Bảng xác suất");
            System.out.println("=".repeat(80));

            System.out.printf("\nXác suất tiên nghiệm:\n");
            System.out.printf("  P(Play=Yes) = %.4f\n", priorYes);
            System.out.printf("  P(Play=No)  = %.4f\n", priorNo);

            String[] attributes = {"Outlook", "Temp", "Humidity", "Windy"};

            for (String attr : attributes) {
                System.out.println("\n" + "-".repeat(80));
                System.out.println("Attribute: " + attr);
                System.out.println("-".repeat(80));

                Map<String, Double> yesProbs = probYes.get(attr);
                Map<String, Double> noProbs = probNo.get(attr);

                Set<String> allValues = new TreeSet<>();
                allValues.addAll(yesProbs.keySet());
                allValues.addAll(noProbs.keySet());

                System.out.printf("%-15s %-20s %-20s\n", "Value", "P(" + attr + "|Yes)", "P(" + attr + "|No)");
                System.out.println("-".repeat(80));

                for (String value : allValues) {
                    double pYes = yesProbs.getOrDefault(value, 0.0);
                    double pNo = noProbs.getOrDefault(value, 0.0);
                    System.out.printf("%-15s %-20.4f %-20.4f\n", value, pYes, pNo);
                }
            }
        }
    }

    public static void main(String[] args) {
        try {
            // Đọc training data
            List<DataPoint> trainingData = readTrainingData("training");
            System.out.println("Cho bộ dữ liệu training có " + trainingData.size() + " mẫu training");
            System.out.println("\nTraining Dataset:");
            System.out.printf("%-4s %-12s %-8s %-12s %-8s %-8s\n",
                    "ID", "Outlook", "Temp", "Humidity", "Windy", "Play");
            System.out.println("-".repeat(80));
            for (int i = 0; i < trainingData.size(); i++) {
                DataPoint p = trainingData.get(i);
                System.out.printf("%-4d %-12s %-8s %-12s %-8s %-8s\n",
                        i + 1, p.outlook, p.temp, p.humidity, p.windy, p.play);
            }

            // Tính xác suất
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Tính xác suất");
            System.out.println("=".repeat(80));

            ProbabilityTable probTable = computeProbabilities(trainingData);
            probTable.print();

            // Đọc và phân loại test data
            List<DataPoint> testData = readTestingData("testing");
            System.out.println("\n\n" + "=".repeat(80));
            System.out.println("Phân loại mẫu testing");
            System.out.println("=".repeat(80));

            for (int i = 0; i < testData.size(); i++) {
                System.out.println("\n" + "-".repeat(80));
                System.out.println("Mẫu test #" + (i + 1));
                System.out.println("-".repeat(80));
                classifyWithNaiveBayes(testData.get(i), probTable);
            }

        } catch (IOException e) {
            System.err.println("Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static BufferedReader getReader(String filename) throws IOException {
        InputStream is = NaiveBayesClassify.class.getClassLoader()
                .getResourceAsStream("ex5/" + filename);

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
            if (parts.length >= 5) {
                data.add(new DataPoint(parts[0], parts[1], parts[2], parts[3], parts[4]));
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
            if (parts.length >= 4) {
                String play = parts.length >= 5 ? parts[4] : "?";
                data.add(new DataPoint(parts[0], parts[1], parts[2], parts[3], play));
            }
        }

        reader.close();
        return data;
    }

    // Tính xác suất
    public static ProbabilityTable computeProbabilities(List<DataPoint> data) {
        ProbabilityTable table = new ProbabilityTable();

        // Đếm số lượng Yes và No
        int countYes = 0;
        int countNo = 0;

        // Đếm tần suất cho mỗi attribute value
        Map<String, Map<String, Integer>> countYes_map = new HashMap<>();
        Map<String, Map<String, Integer>> countNo_map = new HashMap<>();

        String[] attributes = {"Outlook", "Temp", "Humidity", "Windy"};
        for (String attr : attributes) {
            countYes_map.put(attr, new HashMap<>());
            countNo_map.put(attr, new HashMap<>());
        }

        // Đếm
        for (DataPoint point : data) {
            if (point.play.equals("Yes")) {
                countYes++;
                incrementCount(countYes_map.get("Outlook"), point.outlook);
                incrementCount(countYes_map.get("Temp"), point.temp);
                incrementCount(countYes_map.get("Humidity"), point.humidity);
                incrementCount(countYes_map.get("Windy"), point.windy);
            } else if (point.play.equals("No")) {
                countNo++;
                incrementCount(countNo_map.get("Outlook"), point.outlook);
                incrementCount(countNo_map.get("Temp"), point.temp);
                incrementCount(countNo_map.get("Humidity"), point.humidity);
                incrementCount(countNo_map.get("Windy"), point.windy);
            }
        }

        // Tính xác suất tiên nghiệm
        int total = data.size();
        table.priorYes = (double) countYes / total;
        table.priorNo = (double) countNo / total;

        System.out.println("\nĐếm:");
        System.out.println("  Tổng số mẫu: " + total);
        System.out.println("  Play=Yes: " + countYes);
        System.out.println("  Play=No: " + countNo);

        // Tính xác suất có điều kiện
        for (String attr : attributes) {
            Map<String, Double> yesProbs = new HashMap<>();
            Map<String, Double> noProbs = new HashMap<>();

            for (Map.Entry<String, Integer> entry : countYes_map.get(attr).entrySet()) {
                yesProbs.put(entry.getKey(), (double) entry.getValue() / countYes);
            }

            for (Map.Entry<String, Integer> entry : countNo_map.get(attr).entrySet()) {
                noProbs.put(entry.getKey(), (double) entry.getValue() / countNo);
            }

            table.probYes.put(attr, yesProbs);
            table.probNo.put(attr, noProbs);
        }

        return table;
    }

    // Đếm số lần feature xuất hiện trong từng lớp
    private static void incrementCount(Map<String, Integer> map, String key) {
        map.put(key, map.getOrDefault(key, 0) + 1);
    }

    // Phân lớp mẫu dữ liệu dựa trên bảng xác suất đã tính
    public static void classifyWithNaiveBayes(DataPoint testPoint, ProbabilityTable table) {
        System.out.printf("\nMẫu test: Outlook = %s, Temp = %s, Humidity = %s, Windy = %s\n",
                testPoint.outlook, testPoint.temp, testPoint.humidity, testPoint.windy);

        // Tính P(X|Yes) * P(Yes)
        double probYes = table.priorYes;
        System.out.println("\n* P(X|Yes) * P(Yes):");
        System.out.printf("  P(Yes) = %.4f\n", probYes);

        double pOutlookYes = table.probYes.get("Outlook").getOrDefault(testPoint.outlook, 0.0);
        System.out.printf("  P(Outlook=%s|Yes) = %.4f\n", testPoint.outlook, pOutlookYes);
        probYes *= pOutlookYes;

        double pTempYes = table.probYes.get("Temp").getOrDefault(testPoint.temp, 0.0);
        System.out.printf("  P(Temp=%s|Yes) = %.4f\n", testPoint.temp, pTempYes);
        probYes *= pTempYes;

        double pHumidityYes = table.probYes.get("Humidity").getOrDefault(testPoint.humidity, 0.0);
        System.out.printf("  P(Humidity=%s|Yes) = %.4f\n", testPoint.humidity, pHumidityYes);
        probYes *= pHumidityYes;

        double pWindyYes = table.probYes.get("Windy").getOrDefault(testPoint.windy, 0.0);
        System.out.printf("  P(Windy=%s|Yes) = %.4f\n", testPoint.windy, pWindyYes);
        probYes *= pWindyYes;

        System.out.printf("  Kết quả: %.4f × %.4f × %.4f × %.4f × %.4f = %.8f\n",
                table.priorYes, pOutlookYes, pTempYes, pHumidityYes, pWindyYes, probYes);

        // Tính P(X|No) * P(No)
        double probNo = table.priorNo;
        System.out.println("\n* P(X|No) * P(No):");
        System.out.printf("  P(No) = %.4f\n", probNo);

        double pOutlookNo = table.probNo.get("Outlook").getOrDefault(testPoint.outlook, 0.0);
        System.out.printf("  P(Outlook=%s|No) = %.4f\n", testPoint.outlook, pOutlookNo);
        probNo *= pOutlookNo;

        double pTempNo = table.probNo.get("Temp").getOrDefault(testPoint.temp, 0.0);
        System.out.printf("  P(Temp=%s|No) = %.4f\n", testPoint.temp, pTempNo);
        probNo *= pTempNo;

        double pHumidityNo = table.probNo.get("Humidity").getOrDefault(testPoint.humidity, 0.0);
        System.out.printf("  P(Humidity=%s|No) = %.4f\n", testPoint.humidity, pHumidityNo);
        probNo *= pHumidityNo;

        double pWindyNo = table.probNo.get("Windy").getOrDefault(testPoint.windy, 0.0);
        System.out.printf("  P(Windy=%s|No) = %.4f\n", testPoint.windy, pWindyNo);
        probNo *= pWindyNo;

        System.out.printf(" Kết quả: %.4f × %.4f × %.4f × %.4f × %.4f = %.8f\n",
                table.priorNo, pOutlookNo, pTempNo, pHumidityNo, pWindyNo, probNo);

        // Kết luận
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Kết quả cuối:");
        System.out.printf("  P(X|Yes) * P(Yes) = %.8f\n", probYes);
        System.out.printf("  P(X|No)  * P(No)  = %.8f\n", probNo);

        String predicted = probYes > probNo ? "Yes" : "No";
        System.out.printf("\n  → Dự đoán: Play = %s\n", predicted);

        if (!testPoint.play.equals("?")) {
            System.out.printf("  → Thực tế: Play = %s\n", testPoint.play);
            System.out.printf("  → Kết luận: %s\n",
                    predicted.equals(testPoint.play) ? "Đúng" : "Sai");
        }
    }
}
