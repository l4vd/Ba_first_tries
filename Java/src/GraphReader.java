import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.nio.gexf.GEXFImporter;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class GraphReader {

    public static void main(String[] args) {
        // Create a new instance of the GEXFImporter
        GEXFImporter<String, DefaultEdge> importer = new GEXFImporter<>();

        // Create an empty undirected graph
        Graph<String, DefaultEdge> graph = new SimpleGraph<>(DefaultEdge.class);

        // Define the file path of the GEXF file
        String filePath = "path/to/your/graph.gexf";

        try {
            // Import the graph from the GEXF file
            importer.importGraph(graph, new File(filePath));
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + filePath);
            e.printStackTrace();
            return;
        }

        // Print out the vertices and edges of the graph
        System.out.println("Vertices: " + graph.vertexSet());
        System.out.println("Edges: " + graph.edgeSet());
    }
}
