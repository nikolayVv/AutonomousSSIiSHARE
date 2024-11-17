import networkx as nx
import nltk
import json
import os
import numpy as np
import plotly.express as px
import umap
import logging
import torch
import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from utils.helper_functions import open_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch_geometric.utils import from_networkx
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
from itertools import product
from torch_geometric.nn import SAGEConv
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.utils import sort_edge_index
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

nltk.download('wordnet')
nltk.download('wordnet_ic')

brown_ic = wordnet_ic.ic('ic-brown.dat')


class GraphSAGE(torch.nn.Module):
    """The GraphSAGE model used to generate the embeddings."""

    def __init__(self: "GraphSAGE", in_channels: int, out_channels: int, hidden_dim: int = 128, aggregation: str = 'mean', dropout: float = 0.0, num_layers: int = 2) -> None:
        """Initialization of the model.

        Args:
            in_channels (int): The input channels of the input layer.
            out_channels (int): The output channels of the network.
            hidden_dim (int): The hidden dimension channels between the layers.
            aggregation (str): The aggregation function used in each layer.
            dropout (float): The dropout rate applied in each layer.
            num_layers (int): Number of layers in the model.

        Returns:
            None

        """
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(SAGEConv(in_channels, hidden_dim, aggr=aggregation))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))

        # Output layer
        self.layers.append(SAGEConv(hidden_dim, out_channels, aggr=aggregation))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = self.layers[-1](x, edge_index)
        return x


def get_synsets(word: str) -> list:
    """Returns synsets (senses) for a word.

    Args:
        word (str): The word for which the synsets will be generated.

    Returns:
        list: The synsets of the chosen word.

    """
    return wn.synsets(word)


def generate_contextual_embeddings(nx_graph: nx.Graph, max_features: int = 100) -> list[float]:
    """Generate the contextual embeddings based on the description.

    Args:
        nx_graph (Graph): The graph on which we will apply the embeddings.
        max_features (int): The maximum amount of features that will be considered in the embedding.

    Returns:
        list: The generated embedding.

    """
    descriptions = []

    for node, data in nx_graph.nodes(data=True):
        descriptions.append(f"The {data['type']} with title '{node}' has the following description: \"{data['description']}\"")

    vectorizer = TfidfVectorizer(max_features=max_features)
    text_embeddings = vectorizer.fit_transform(descriptions).toarray()
    return text_embeddings


def execute_evaluation(dimensions, hidden_dim, dropout, aggregation, num_layers, max_features, data: any, subset_properties, k_nearest=5):
    """Executing evaluation on a subset of properties:

    Args:
        dimensions (int): The out channels of the model.
        hidden_dim (int): The hidden channels of the model.
        dropout (float): The dropout applied in the model.
        aggregation (str): The aggregation applied in the model.
        num_layers (int): The number of layers in the model.
        max_features (int): The number of features.
        data (generator): Generator containing the data of the graph.
        subset_properties (list): The properties, which will be evaluated/aligned.
        k_nearest (int): The nearest/most similar properties we will be looking for.

    Returns:
        None

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    properties = [node for node, data in nx_graph.nodes(data=True) if data.get('type') == "property"]
    all_nodes = list(nx_graph.nodes())

    text_embeddings = generate_contextual_embeddings(nx_graph, max_features=max_features)
    data.x = torch.tensor(text_embeddings, dtype=torch.float)

    model = GraphSAGE(
        in_channels=text_embeddings.shape[1],
        out_channels=dimensions,
        hidden_dim=hidden_dim,
        aggregation=aggregation,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        try:
            graphsage_embeddings = model(data).cpu().numpy()
        except Exception as e:
            logging.error(
                f'dim{dimensions}_hiddim{hidden_dim}_drop{dropout}_{aggregation}_lay{num_layers}_feat{max_features}: {e}')

    property_embeddings_dict = {prop: graphsage_embeddings[idx] for idx, prop in enumerate(properties)}

    # Extract embeddings for the subset of properties
    subset_embeddings = np.array([property_embeddings_dict[prop] for prop in subset_properties])

    # Compute cosine similarity between the selected subset of properties and all properties
    all_property_embeddings = np.array(list(property_embeddings_dict.values()))
    similarity_matrix = cosine_similarity(subset_embeddings, all_property_embeddings)

    # For each property in subset, find k-nearest neighbors and log results
    nearest_neighbors = {}
    property_class_associations = {}

    for i, prop in enumerate(subset_properties):
        # Find k-nearest neighbors
        similarities = similarity_matrix[i]
        nearest_indices = similarities.argsort()[-k_nearest - 1:-1][::-1]  # Top k nearest, excluding itself
        nearest_props = [properties[idx] for idx in nearest_indices]
        nearest_neighbors[prop] = nearest_props

        # Find classes associated with the property
        associated_classes = [
            neighbor for neighbor in nx_graph.neighbors(prop)
            if nx_graph.get_edge_data(prop, neighbor).get("label") == "propertyOf"
        ]
        property_class_associations[prop] = associated_classes

        # Log information for human evaluation
        logging.info(f"Property: {prop}, Nearest Neighbors: {nearest_props}, Associated Classes: {associated_classes}")

    # Visualization setup
    G = nx.Graph()

    # Add nodes and edges for each property and its neighbors
    for prop, neighbors in nearest_neighbors.items():
        # Add the main property node
        G.add_node(prop, label=prop, classes=property_class_associations[prop])

        # Add each neighbor and an edge between the property and its neighbor
        for neighbor in neighbors:
            # Find classes associated with the neighbor
            neighbor_classes = [
                neighbor_class for neighbor_class in nx_graph.neighbors(neighbor)
                if nx_graph.get_edge_data(neighbor, neighbor_class).get("label") == "propertyOf"
            ]
            G.add_node(neighbor, label=neighbor, classes=neighbor_classes)
            G.add_edge(prop, neighbor)

    # Prepare Plotly visualization
    pos = nx.spring_layout(G, seed=42)

    # Create Plotly scatter plot for nodes
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        classes = G.nodes[node]['classes']
        node_text.append(f"{G.nodes[node]['label']}<br>Schemas: {', '.join(classes) if classes else 'None'}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Class Count',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Property and Nearest Neighbor Relationships with Associated Classes',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',  # Set the plot background color to white
                        paper_bgcolor='white'
                    ),
                    )
    fig.show()


def generate_graph(classes: dict[str, dict], properties: dict[str, dict]) -> nx.Graph:
    """Generate the graph based on the classes and properties.

    Args:
        classes (dict):A dictionary containing the classes in our ontology.
        properties (dict): A dictionary containing the properties in our ontology.

    Returns:
        Graph: The generated graph.

    """
    nx_graph = nx.Graph()

    for class_name, class_info in classes.items():
        if "description" not in class_info:
            synsets = get_synsets(class_info["title"])
            class_info["description"] = f"A class called '{class_info['title']}'. "
            if synsets:
                class_info["description"] = class_info["description"] + synsets[0].definition()

        nx_graph.add_node(class_name, type="class", description=class_info["description"])

        for parent_class in class_info.get("subClassOf", []):
            if "owl#" in parent_class:
                parent_class = parent_class.split("#")[1]
            if parent_class in classes:
                nx_graph.add_edge(class_name, parent_class, label='subClassOf')
            else:
                logging.warning(f"Domain class '{parent_class}' from class '{class_name}' not a class.")

    for prop_name, prop_info in properties.items():
        if "description" not in prop_info:
            synsets = get_synsets(prop_info["title"])
            prop_info[
                "description"] = f"A property called '{prop_info['title']}' which is part of the classes '{''.join(prop_info.get('classes', []))}'. "
            if synsets:
                prop_info["description"] = prop_info["description"] + synsets[0].definition()

        nx_graph.add_node(prop_name, type="property", description=prop_info["description"])

        for domain_class in prop_info.get("classes", []):
            if "owl#" in domain_class:
                domain_class = domain_class.split("#")[1]
            if domain_class in classes or domain_class in properties:
                nx_graph.add_edge(prop_name, domain_class, label='propertyOf')
            else:
                logging.warning(f"Domain class '{domain_class}' from property '{prop_name}' not a class or property.")

        for parent_property in prop_info.get("subPropertyOf", []):
            if "owl#" in parent_property:
                parent_property = parent_property.split("#")[1]
            if parent_property in properties or parent_property in classes:
                nx_graph.add_edge(prop_name, parent_property, label='subPropertyOf')
            else:
                logging.warning(f"Property '{parent_property}' from property '{prop_name}' not a property or class.")

        for type_property in prop_info.get("types", []):
            if "owl#" in type_property:
                type_property = type_property.split("#")[1]
            if type_property in properties or type_property in classes:
                nx_graph.add_edge(prop_name, type_property, label='type')
            else:
                logging.warning(f"Type property '{type_property}' from property '{prop_name}' not a property or class.")

    return nx_graph


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"building_ontology_logging.log"),
            logging.StreamHandler(),
        ],
    )

    data_dir = "./data/merged"
    classes, properties = open_dataset(data_dir)

    # Convert the ontology into a NetworkX graph
    program_type = "fine-tune"
    nx_graph = generate_graph(classes, properties)

    data = from_networkx(nx_graph)
    data.edge_index = sort_edge_index(data.edge_index, sort_by_row=False)

    subset_properties = [
        "payload",
        "roofLoad",
        "weightTotal",
        "employee",
        "product",
        "maximumIntake",
        "activeIngredient"
    ]
    if program_type == "fine-tune":
        execute_evaluation(128, 256, 0.1, "max", 10, 300, data, subset_properties)
    else:
        # Parameters searching
        dimensions_list = [32, 64, 128]  # GraphSAGE output dimensions
        hidden_dim_list = [64, 128, 256]  # Hidden layer dimensions in GraphSAGE
        dropout_list = [0.1, 0.3, 0.5]  # Dropout rates
        aggregation_list = ['mean', 'max']  # Aggregation methods in GraphSAGE
        num_layers_list = [2, 5, 10]  # Number of layers in the GraphSAGE network
        max_features_list = [100, 200, 300]  # Number of TF-IDF features
        n_clusters_list = [3, 5, 10, 20, 30, 40, 50, 75, 100]
        pca_components = 2

        combinations = list(product(
            dimensions_list,
            hidden_dim_list,
            dropout_list,
            aggregation_list,
            num_layers_list,
            max_features_list,
        ))
        os.makedirs('./experiments', exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nodes = [node for node, data in nx_graph.nodes(data=True) if data.get('type') == "property"]
        all_nodes = list(nx_graph.nodes())
        results = []
        clusters_data = {"node_labels": nodes}

        for dimensions, hidden_dim, dropout, aggregation, num_layers, max_features in tqdm(combinations, desc="Best-param search"):
            text_embeddings = generate_contextual_embeddings(nx_graph, max_features=max_features)
            data.x = torch.tensor(text_embeddings, dtype=torch.float)

            # Initialize the GraphSAGE model with current parameters
            model = GraphSAGE(
                in_channels=text_embeddings.shape[1],
                out_channels=dimensions,
                hidden_dim=hidden_dim,
                aggregation=aggregation,
                dropout=dropout,
                num_layers=num_layers,
            ).to(device)
            data = data.to(device)

            model.eval()
            with torch.no_grad():
                try:
                    graphsage_embeddings = model(data).cpu().numpy()
                except Exception as e:
                    logging.error(f'dim{dimensions}_hiddim{hidden_dim}_drop{dropout}_{aggregation}_lay{num_layers}_feat{max_features}: {e}')

            property_indices = [all_nodes.index(node) for node in nodes]  # Get the indices of the property nodes
            property_embeddings = graphsage_embeddings[property_indices]

            pca = PCA(n_components=pca_components)
            reduced_embeddings = pca.fit_transform(graphsage_embeddings)

            for n_clusters in n_clusters_list:
                # Apply K-Means clustering with current parameters
                kmeans = KMeans(n_clusters=n_clusters)
                labels = kmeans.fit_predict(graphsage_embeddings)

                # Calculate evaluation metrics
                silhouette_avg = silhouette_score(graphsage_embeddings, labels)
                calinski_harabasz_avg = calinski_harabasz_score(graphsage_embeddings, labels)
                davies_bouldin_avg = davies_bouldin_score(graphsage_embeddings, labels)
                inertia = kmeans.inertia_

                # Save the result for comparison
                results.append({
                    'dimensions': int(dimensions),
                    'hidden_dim': int(hidden_dim),
                    'dropout': float(dropout),
                    'aggregation': aggregation,
                    'num_layers': int(num_layers),
                    'max_features': int(max_features),
                    'n_clusters': int(n_clusters),
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz_avg),
                    'davies_bouldin_score': float(davies_bouldin_avg),
                    'inertia': float(inertia),
                })

                clusters_data[f'{dimensions}_{hidden_dim}_{dropout}_{aggregation}_{num_layers}_{max_features}_{n_clusters}'] = labels.tolist()

        os.makedirs('./experiments', exist_ok=True)
        with open(f'./experiments/graph_results.json', 'w') as f:
            json.dump(results, f)

        with open(f'./experiments/graph_clusters.json', 'w') as f:
            json.dump(clusters_data, f)