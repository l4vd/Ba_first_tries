import org.gephi.graph.api.*;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.EdgeDirectionDefault;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.gephi.statistics.plugin.*;
import org.openide.util.Lookup;

import java.io.File;

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


        // Calculate Degree
        Degree degree = new Degree();
        degree.execute(graphModel);
        System.out.println("Average Degree: " + degree.getAverageDegree());

        // Calculate Weighted Degree
        WeightedDegree weightedDegree = new WeightedDegree();
        weightedDegree.execute(graphModel);
        //System.out.println("Average Weighted Degree: " + weightedDegree.getAverageWeightedDegree());

        /*
        // Calculate Eccentricity
        Eccentricity eccentricity = new Eccentricity();
        eccentricity.execute(graphModel);

        // Calculate Closeness Centrality
        ClosenessCentrality closenessCentrality = new ClosenessCentrality();
        closenessCentrality.execute(graphModel);

        // Calculate Harmonic Closeness Centrality
        HarmonicClosenessCentrality harmonicClosenessCentrality = new HarmonicClosenessCentrality();
        harmonicClosenessCentrality.execute(graphModel);

        // Calculate Betweenness Centrality
        BetweennessCentrality betweennessCentrality = new BetweennessCentrality();
        betweennessCentrality.execute(graphModel); */

        // Calculate Modularity Class
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

        /*Graph graph = graphModel.getGraph();

        for (Node node : graph.getNodes()) {
            System.out.println("Node ID: " + node.getId());

            // Print node attributes
            for (Column col : graphModel.getNodeTable()) {
                System.out.println(col.getTitle() + ": " + node.getAttribute(col));
            }

            // Print calculated metrics
            System.out.println("Degree: " + degree.getDegree(graph, node));
            System.out.println("Weighted Degree: " + weightedDegree.getDegree(graph, node));
            System.out.println("Eccentricity: " + eccentricity.getEccentricity(graph, node));
            System.out.println("Closeness Centrality: " + closenessCentrality.getCentrality(graph, node));
            System.out.println("Harmonic Closeness Centrality: " + harmonicClosenessCentrality.getCentrality(graph, node));
            System.out.println("Betweenness Centrality: " + betweennessCentrality.getCentrality(graph, node));
            System.out.println("Modularity Class: " + modularity.getModularityClass(node));
            System.out.println("Clustering Coefficient: " + clusteringCoefficient.getClusteringCoefficient(graph, node));
            System.out.println("Eigenvector Centrality: " + eigenvectorCentrality.getCentrality(graph, node));
            System.out.println();
        }*/

    }
}
