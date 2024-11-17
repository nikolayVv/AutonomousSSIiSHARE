import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.colors as mcolors


def plot(properties: bool = False) -> tuple[nx.DiGraph, list[str]]:
    """Function that plots the merged ontology from the generated properties or classes

    Args:
        properties (bool): defines if we are using properties or not

    Returns:
        tuple: containing the graph of classes/properties and the corresponding colors

    """
    ontology_combinations_color_map = {
        ("Schema ORG",): "red",
        ("DBpedia",): "blue",
        ("YAGO",): "green",
        ("DBpedia", "Schema ORG"): "purple",
        ("Schema ORG", "YAGO"): "orange",
        ("DBpedia", "YAGO"): "cyan",
        ("DBpedia", "Schema ORG", "YAGO"): "brown",
        (): "gray"
    }

    if properties:
        with open("../data/merged/properties.json", "r") as f:
            data = json.load(f)
    else:
        with open("../data/merged/classes.json", "r") as f:
            data = json.load(f)

    G = nx.DiGraph()
    node_colors = []

    for class_name, class_info in data.items():
        G.add_node(class_name)

        attribute = "subClassOf"
        if properties:
            attribute = "subPropertyOf"

        if attribute in class_info:
            for parent_class in class_info[attribute]:
                if parent_class in data:
                    G.add_edge(parent_class, class_name)

        if "ontology" in class_info:
            ontologies = tuple(sorted(class_info["ontology"]))  # Sort and convert to tuple for consistency
            class_colors = ontology_combinations_color_map.get(ontologies, "gray")  # Use default color if not in map
            if (ontology_combinations_color_map.get(ontologies, "gray")) == "gray":
                print(ontologies)
            node_colors.append(class_colors)
        else:
            print("NO")
            node_colors.append("gray")

    return G, node_colors


if __name__ == "__main__":
    G, colors = plot()

    # Calculate node degrees
    node_degrees = dict(G.degree())

    # Normalize node sizes based on degree (adjust scaling factor as needed)
    node_sizes = [node_degrees[node] * 50 for node in G.nodes()]

    # Create a dictionary for font sizes (use degree-based scaling)
    font_sizes = {node: max(8, node_degrees[node] * 0.5) for node in G.nodes()}

    # Create layout
    pos = nx.spring_layout(G, k=1.15, iterations=50)
    degree_threshold = 3

    plt.figure(figsize=(20, 20))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.5)

    # Only draw labels for nodes with high degrees
    for node, (x, y) in pos.items():
        if node_degrees[node] >= degree_threshold:
            plt.text(x, y, s=node, fontsize=font_sizes[node]*0.5, ha='center', va='center', color='black')

    red_patch = mpatches.Patch(color='red', label='Schema ORG')
    blue_patch = mpatches.Patch(color='blue', label='DBpedia')
    green_patch = mpatches.Patch(color='green', label='YAGO')
    purple_patch = mpatches.Patch(color='purple', label='Schema ORG + DBpedia')
    orange_patch = mpatches.Patch(color='orange', label='Schema ORG + YAGO')
    cyan_patch = mpatches.Patch(color='cyan', label='DBpedia + YAGO')
    brown_patch = mpatches.Patch(color='brown', label='Schema ORG + DBpedia + YAGO')
    plt.legend(handles=[blue_patch, blue_patch, green_patch, purple_patch, orange_patch, cyan_patch, brown_patch], loc='upper left')

    plt.axis('off')
    plt.show()