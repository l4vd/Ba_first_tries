import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphModel;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.EdgeDirectionDefault;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
//import org.jgrapht.nio.gexf.GEXFImporter;
import org.openide.util.Lookup;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

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
        System.out.println("Edges: " + graph.edgeSet());*/
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
    }
}
