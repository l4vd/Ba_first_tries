import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.AgnosticEncoders;
import org.gephi.graph.api.Graph;
import scala.collection.JavaConverters;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

public class Test {

    public static void main(String[] args) {
        // Create SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("Read CSV with Spark")
                .master("local[*]")  // Specify Spark master
                .getOrCreate();

        // Read CSV file into DataFrame
        Dataset<Row> df_artists = spark
                .read()
                .option("header", "true")  // Use first line of file as header
                .option("inferSchema", "true")  // Automatically infer data types
                .csv("C:/tmp/a/Bachelor_Research/Ba_first_tries/Control_Network_Creation/Kontrolldaten/Artists.csv");

        Dataset<Row> df_songs = spark
                .read()
                .option("header", "true")  // Use first line of file as header
                .option("inferSchema", "true")  // Automatically infer data types
                .csv("C:/tmp/a/Bachelor_Research/Ba_first_tries/Control_Network_Creation/Kontrolldaten/Songs.csv");

        // Process DataFrame df_artists and create nodes
        df_artists.foreach((ForeachFunction<Row>) row -> {
            String spotifyId = row.getString(row.fieldIndex("Spotify ID"));
            scala.collection.immutable.Map<String, Object> scalaMap = row.getValuesMap(JavaConverters.asScalaBufferConverter(Arrays.asList(row.schema().fieldNames())).asScala().toSeq());
            Map<String, Object> nodeAttrs = JavaConverters.mapAsJavaMapConverter(scalaMap).asJava();
            // Create node with attributes in Gephi-compatible format
            createNodeInGephi(spotifyId, nodeAttrs);
        });

        // Process DataFrame df_songs and create edges
        df_songs.foreach((ForeachFunction<Row>) row -> {
            String label1 = row.getString(row.fieldIndex("label1"));
            String label2 = row.getString(row.fieldIndex("label2"));
            scala.collection.immutable.Map<String, Object> scalaMap = row.getValuesMap(JavaConverters.asScalaBufferConverter(Arrays.asList(row.schema().fieldNames())).asScala().toSeq());
            Map<String, Object> edgeAttrs = JavaConverters.mapAsJavaMapConverter(scalaMap).asJava();
            // Create edge with attributes in Gephi-compatible format
            createEdgeInGephi(label1, label2, edgeAttrs);
        });

        // Show DataFrame contents
        df_artists.show();

        // Stop SparkSession
        spark.stop();
    }

    // Function to create node in Gephi (write to GEXF file)
    private static void createNodeInGephi(String nodeId, Map<String, Object> attributes) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("nodes.gexf", true))) {
            writer.write("<node id=\"" + nodeId + "\">");
            for (Map.Entry<String, Object> entry : attributes.entrySet()) {
                writer.write("<attvalue for=\"" + entry.getKey() + "\" value=\"" + entry.getValue() + "\"/>");
            }
            writer.write("</node>");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Function to create edge in Gephi (write to GEXF file)
    private static void createEdgeInGephi(String sourceNodeId, String targetNodeId, Map<String, Object> attributes) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("edges.gexf", true))) {
            writer.write("<edge source=\"" + sourceNodeId + "\" target=\"" + targetNodeId + "\">");
            for (Map.Entry<String, Object> entry : attributes.entrySet()) {
                writer.write("<attvalue for=\"" + entry.getKey() + "\" value=\"" + entry.getValue() + "\"/>");
            }
            writer.write("</edge>");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
