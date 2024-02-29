import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class DataFrame {
    /*private List<String[]> data;
    private String[] headers;

    public DataFrame(List<String[]> data, String[] headers) {
        this.data = data;
        this.headers = headers;
    }

    public List<String[]> getData() {
        return data;
    }

    public String[] getHeaders() {
        return headers;
    }

    public static void main(String[] args) {
        String filePath = "../../Control_Network_Creation/Kontrolldaten/Artists.csv";
        List<String[]> data = new ArrayList<>();
        String[] headers = null;

        try (CSVParser parser = CSVParser.parse(new File(filePath), CSVFormat.DEFAULT.withHeader())) {
            for (CSVRecord record : parser) {
                if (headers == null) {
                    headers = record.iterator().next().split(",");
                }
                data.add(record.iterator().next().split(","));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataFrame df = new DataFrame(data, headers);

        // Example usage:
        System.out.println("Headers: ");
        for (String header : df.getHeaders()) {
            System.out.print(header + "\t");
        }
        System.out.println("\nData: ");
        for (String[] row : df.getData()) {
            for (String value : row) {
                System.out.print(value + "\t");
            }
            System.out.println();
        }
    }*/
}
