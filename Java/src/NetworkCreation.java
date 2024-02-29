import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.gephi.graph.api.DirectedGraph;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.project.api.ProjectController;
import org.openide.util.Lookup;

public class NetworkCreation {

    /*public static void main(String[] args) throws IOException {
        // Initialize Gephi project
        ProjectController pc = Lookup.getDefault().lookup(ProjectController.class);
        pc.newProject();

        // Get a graph model
        GraphController gc = Lookup.getDefault().lookup(GraphController.class);
        GraphModel gm = gc.getGraphModel();
        DirectedGraph graph = gm.getDirectedGraph();

        // Read Artists.csv
        File artistsFile = new File("../../Control_Network_Creation/Kontrolldaten/Artists.csv");
        CSVParser artistsParser = new CSVParser(new FileReader(artistsFile), CSVFormat.DEFAULT.withHeader());

        // Add nodes with attributes to the graph
        for (CSVRecord record : artistsParser) {
            String spotifyId = record.get("Spotify ID");
            Node node = gm.factory().newNode(spotifyId);
            for (String header : artistsParser.getHeaderNames()) {
                if (!header.equals("Spotify ID")) {
                    node.getAttributes().setValue(header, record.get(header));
                }
            }
            graph.addNode(node);
        }

        // Read Songs.csv
        File songsFile = new File("../../Control_Network_Creation/Kontrolldaten/Songs.csv");
        CSVParser songsParser = new CSVParser(new FileReader(songsFile), CSVFormat.DEFAULT.withHeader());

        // Add edges with attributes to the graph
        for (CSVRecord record : songsParser) {
            String label1 = record.get("label1");
            String label2 = record.get("label2");
            graph.addEdge(gm.factory().newEdge(label1 + "_" + label2, graph.getNode(label1), graph.getNode(label2), 1.0, true));
        }

        // Save the project
        pc.saveProject();
    }*/
}
