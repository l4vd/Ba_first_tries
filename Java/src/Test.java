import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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



        // Show DataFrame contents
        df_artists.show();

        // Stop SparkSession
        spark.stop();
    }
}
