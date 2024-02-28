import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.gephi.graph.api.DirectedGraph;
import org.gephi.graph.api.GraphController;
import org.gephi.io.exporter.api.ExportController;
import org.gephi.io.exporter.preview.PNGExporter;
import org.gephi.project.api.ProjectController;
import org.openide.util.Lookup;

public class NetworkCreation {

    public static void main(String[] args) throws IOException {
        // Initialize Gephi project
        ProjectController pc = Lookup.getDefault().lookup(ProjectController.class);
        pc.newProject();

        // Get a graph model
        GraphController gc = Lookup.getDefault().lookup(GraphController.class);
        DirectedGraph graph = gc.getGraphModel().getDirectedGraph();

        // Read Artists.csv
        File artistsFile = new File("../../Control_Network_Creation/Kontrolldaten/Artists.csv");
        CSVParser artistsParser = new CSVParser(new FileReader(artistsFile), CSVFormat.DEFAULT.withHeader());
        Map<String, String> artistAttributes = new HashMap<>();
        for (CSVRecord record : artistsParser) {
            String spotifyId = record.get("Spotify ID");
            for (String header : artistsParser.getHeaderNames()) {
                if (!header.equals("Spotify ID")) {
                    artistAttributes.put(header, record.get(header));
                }
            }
            graph.addNode(spotifyId).getNodeData().getAttributes().setValue(artistAttributes);
        }

        // Read Songs.csv
        File songsFile = new File("../../Control_Network_Creation/Kontrolldaten/Songs.csv");
        CSVParser songsParser = new CSVParser(new FileReader(songsFile), CSVFormat.DEFAULT.withHeader());
        for (CSVRecord record : songsParser) {
            String label1 = record.get("label1");
            String label2 = record.get("label2");
            graph.addEdge(label1 + "_" +
