import org.gephi.graph.api.*;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.EdgeDirectionDefault;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.gephi.statistics.plugin.*;
import org.openide.util.Lookup;
import org.gephi.io.exporter.api.ExportController;
import org.gephi.io.exporter.spi.Exporter;
import org.openide.util.Lookup;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class GraphReader {

    public static void main(String[] args) {
        /*// Create a new instance of the GEXFImporter
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
        System.out.println("Edges: " + graph.edgeSet());K*/
    }
}


class Main {
    public static void main(String[] args) {
        // Initialize the Gephi toolkit
        ProjectController pc = Lookup.getDefault().lookup(ProjectController.class);
        pc.newProject();
        Workspace workspace = pc.getCurrentWorkspace();

        // Import file
        ImportController importController = Lookup.getDefault().lookup(ImportController.class);
        Container container;
        try {
            File file = new File("C:/tmp/a/Bachelor_Research/Ba_first_tries/Control_Network_Creation/network/Successful Network/Original Network/original_network.gexf");
            container = importController.importFile(file);
            container.getLoader().setEdgeDefault(EdgeDirectionDefault.UNDIRECTED);
        } catch (Exception ex) {
            ex.printStackTrace();
            return;
        }

        // Append imported data to GraphAPI
        importController.process(container, new DefaultProcessor(), workspace);

        // Get a graph model - it exists because we have a workspace
        GraphModel graphModel = Lookup.getDefault().lookup(GraphController.class).getGraphModel();

        System.out.println("Nodes: " + graphModel.getGraph().getNodeCount());
        System.out.println("Edges: " + graphModel.getGraph().getEdgeCount());

        // Assuming you have the GraphModel as 'graphModel'
        Graph graph = graphModel.getGraph();

        calcAndPrintAvgMetrics(graphModel, graph);

        // Initialize the GraphDistance algorithm
        GraphDistance distance = new GraphDistance();
        distance.setDirected(false);  // Set to true if your graph is directed

        // Execute the algorithm
        distance.execute(graphModel);

        // Get the ExportController
        ExportController ec = Lookup.getDefault().lookup(ExportController.class);

        // Export the graph to a file
        try {
            // Export nodes
            Exporter exporter = ec.getExporter("csv"); // Get CSV exporter
            exporter.setWorkspace(workspace);
            ec.exportFile(new File("Java/resources/nodes.csv"), exporter);

            // Export edges
            ec.exportFile(new File("Java/resources/edges.csv"), exporter);
        } catch (IOException ex) {
            ex.printStackTrace();
            return;
        }

    }

    private static void calcAndPrintAvgMetrics(GraphModel graphModel, Graph graph) {
        // Calculate Degree
        Degree degree = new Degree();
        degree.execute(graphModel);
        System.out.println("Average Degree: " + degree.getAverageDegree());

        // Calculate Weighted Degree
        WeightedDegree weightedDegree = new WeightedDegree();
        weightedDegree.execute(graphModel);
        System.out.println("Average Weighted Degree: " + weightedDegree.calculateAverageWeightedDegree(graph, false, false));

        // Calculate Modularity
        Modularity modularity = new Modularity();
        modularity.execute(graphModel);
        System.out.println("Modularity: " + modularity.getModularity());

        // Calculate Clustering Coefficient
        ClusteringCoefficient clusteringCoefficient = new ClusteringCoefficient();
        clusteringCoefficient.execute(graphModel);
        System.out.println("Average Clustering Coefficient: " + clusteringCoefficient.getAverageClusteringCoefficient());

        // Calculate Eigenvector Centrality
        EigenvectorCentrality eigenvectorCentrality = new EigenvectorCentrality();
        eigenvectorCentrality.execute(graphModel);
        //System.out.println("Average Eigenvector Centrality: " + eigenvectorCentrality.getAverageCentrality());
    }
}
