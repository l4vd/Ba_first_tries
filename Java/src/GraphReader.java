import org.gephi.graph.api.*;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.EdgeDirectionDefault;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.gephi.statistics.plugin.*;
import org.openide.util.Lookup;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;

class GraphReader {
    public static void main(String[] args) {
        // Initialize the Gephi toolkit
        ProjectController pc = Lookup.getDefault().lookup(ProjectController.class);
        pc.newProject();
        Workspace workspace = pc.getCurrentWorkspace();

        // Import file
        ImportController importController = Lookup.getDefault().lookup(ImportController.class);
        Container container;
        try {
            File file = new File("C:/tmp/a/Bachelor_Research/Ba_first_tries/Control_Network_Creation/Kontrolldaten/Kontrolldaten_Graph.gexf");
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

        String filePath = "Java/resources/nodes.csv";
        exportNodeList(graph, filePath);

        filePath = "Java/resources/edges.csv";
        exportEdgeList(graph, filePath);
    }

    public static void exportNodeList(Graph graph, String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            boolean firstIt = true;
            for (Node node : graph.getNodes()) {
                Set<String> attributeKeys = node.getAttributeKeys();
                if (firstIt) {
                    writer.write("Id");
                    for (String attribute : attributeKeys
                    ) {
                        writer.write("\t" + attribute);
                    }
                    writer.write(("\n"));
                    firstIt = false;
                }
                writer.write(node.getId() + "");
                for (String attribute : attributeKeys
                ) {
                    writer.write("\t" + node.getAttribute(attribute));
                }
                writer.write(("\n"));
            }

            System.out.println("Node list exported to " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void exportEdgeList(Graph graph, String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            boolean firstIt = true;
            for (Edge edge : graph.getEdges()) {
                Node source = edge.getSource();
                Node target = edge.getTarget();
                Set<String> attributeKeys = edge.getAttributeKeys();
                if (firstIt) {
                    writer.write("Source\tTarget\tType\tWeight"); // Assuming you want to export Source, Target, Type, and Weight
                    for (String attribute : attributeKeys
                    ) {
                        writer.write("\t" + attribute);
                    }
                    writer.write(("\n"));
                    firstIt = false;
                }
                writer.write(source.getId() + "\t" + target.getId() + "\t" + edge.getType() + "\t" + edge.getWeight());
                for (String attribute : attributeKeys
                ) {
                    writer.write("\t" + edge.getAttribute(attribute));
                }
                writer.write(("\n"));
            }
            System.out.println("Edge list exported to " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
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
    }
}
