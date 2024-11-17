import nltk
import numpy as np
import json
import os
import logging
import random
import plotly.graph_objects as go
import networkx as nx
import torch

from utils.helper_functions import open_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
from itertools import product
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, DebertaTokenizer, DebertaModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')
nltk.download('wordnet_ic')

brown_ic = wordnet_ic.ic('ic-brown.dat')

def get_synsets(word: str) -> list:
    """Returns synsets (senses) for a word.

    Args:
        word (str): The word for which the synsets will be generated.

    Returns:
        list: The synsets of the chosen word.

    """
    return wn.synsets(word)


def generate_embedding(text: str, model: any, tokenizer: any, device: str ='cpu') -> list[float]:
    """Generate contextual embedding of a string using the chosen model and tokenizer.

    Args:
        text (str): The string, which will be embedded.
        model (Model): The model that will be used to generate the embedding.
        tokenizer (Tokenizer): The tokenizer that will be used to generate the embedding.
        device (str): The device on which the model and the tokenizer will run.

    Returns:
        list: The generated contextual embedding.

    """
    if tokenizer is None:
        embedding = model.encode([text], show_progress_bar=False)[0]
        return embedding
    else:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token embedding
        return cls_embedding[0]


def generate_enhanced_class_embedding(classes: dict[str, dict], curr_class: str, depth: int, init_depth: int, pooling: str, alpha: float) -> list[float]:
    """Generating the enhanced class contextual embedding.

    Args:
        classes (dict): The classes in the ontology and their corresponding contextual embeddings.
        curr_class (str): The name of the current class for which the embedding will be generated.
        depth (int): The contextual depth that will be used for the class.
        init_depth (int): The initial contextual depth that is used for the class.
        pooling (str): The pooling that will be applied to the class embeddings.
        alpha (float): The alpha parameter of the class.

    Returns:
        list: The generated enhanced property embedding.

    """
    curr_embedding = curr_class["embedding"]

    if depth == 0:
        return alpha * curr_embedding

    parent_classes = curr_class.get("subClassOf", [])
    if len(parent_classes) == 0:
        return alpha * curr_embedding

    embeddings = []
    for parent_class in parent_classes:
        embeddings.append(generate_enhanced_class_embedding(
            classes, classes[parent_class], depth - 1, init_depth, pooling, alpha,
        ))

    if pooling == "mean":
        parent_embedding = np.mean(embeddings, axis=0)
    elif pooling == "sum":
        parent_embedding = np.sum(embeddings, axis=0)
    else:
        parent_embedding = np.max(embeddings, axis=0)

    return alpha * curr_embedding + (1 - alpha) * parent_embedding


def generate_enhanced_property_embedding(properties: dict[str, dict], classes: dict[str, dict], curr_property: str, d_p: int, d_c: int, i_d_p: int, p_p: str, p_c: str, a_p: float, a_c: float, b_p: float, g_p: float) -> list[float]:
    """Generating the enhanced property contextual embedding.

    Args:
        properties (dict): The properties in the ontology and their corresponding contextual embeddings.
        classes (dict): The classes in the ontology and their corresponding contextual embeddings.
        curr_property (str): The name of the current property for which the embedding will be generated.
        d_p (int): The contextual depth that will be used for the property.
        d_c (int): The contextual depth that will be used for the class.
        i_d_p (int): The initial contextual depth that is used for the property.
        p_p (str): The pooling that will be applied to the property embeddings.
        p_c (str): The pooling that will be applied to the class embeddings.
        a_p (float): The alpha parameter of the property.
        a_c (float): The alpha parameter of the class.
        b_p (float): The beta parameter of the property.
        g_p (float): The gamma parameter of the property.

    Returns:
        list: The generated enhanced property embedding.

    """
    curr_embedding = curr_property["embedding"]

    if d_p == 0:
        return a_p * curr_embedding

    parent_properties = curr_property.get("subPropertyOf", [])
    property_classes = curr_property.get("classes", [])
    property_types = curr_property.get("types", [])

    if len(parent_properties) == 0:
        parent_embedding = None
    else:
        embeddings = []
        for parent_property in parent_properties:
            if parent_property in classes:
                embeddings.append(generate_enhanced_class_embedding(
                    classes, classes[parent_property], d_c, d_c, p_c, a_c,
                ))
            else:
                embeddings.append(generate_enhanced_property_embedding(
                    properties, classes, properties[parent_property], d_p - 1, d_c, i_d_p, p_p, p_c, a_p, a_c, b_p, g_p,
                ))

        if p_p == "mean":
            parent_embedding = np.mean(embeddings, axis=0)
        elif p_p == "sum":
            parent_embedding = np.sum(embeddings, axis=0)
        else:
            parent_embedding = np.max(embeddings, axis=0)

    if len(property_classes) == 0:
        class_embedding = None
    else:
        embeddings = []
        for property_class in property_classes:
            embeddings.append(generate_enhanced_class_embedding(
                classes, classes[property_class], d_c, d_c, p_c, a_c,
            ))

        if p_p == "mean":
            class_embedding = np.mean(embeddings, axis=0)
        elif p_p == "sum":
            class_embedding = np.sum(embeddings, axis=0)
        else:
            class_embedding = np.max(embeddings, axis=0)

    if len(property_types) == 0:
        type_embedding = None
    else:
        embeddings = []
        for property_type in property_types:
            if property_type in classes:
                embeddings.append(generate_enhanced_class_embedding(
                    classes, classes[property_type], d_c, d_c, p_c, a_c,
                ))
            else:
                embeddings.append(generate_enhanced_property_embedding(
                    properties, classes, properties[property_type], d_p - 1, d_c, i_d_p, p_p, p_c, a_p, a_c, b_p, g_p,
                ))

        if p_p == "mean":
            type_embedding = np.mean(embeddings, axis=0)
        elif p_p == "sum":
            type_embedding = np.sum(embeddings, axis=0)
        else:
            type_embedding = np.max(embeddings, axis=0)

    final_embedding = a_p * curr_embedding
    if parent_embedding is not None:
        final_embedding += b_p * parent_embedding

    if class_embedding is not None:
        final_embedding += g_p * class_embedding

    if type_embedding is not None:
        final_embedding += (1 - a_p - b_p - g_p) * type_embedding

    return final_embedding / np.linalg.norm(final_embedding)


def execute_evaluation(depth_class: int, depth_property: int, alpha_class: float, alpha_prop: float, beta: float, gamma: float, pooling_class: str, pooling_property: str, embedded_properties: dict, embedded_classes: dict, subset_properties: list[str], model_name: str, k_nearest: int = 5) -> None:
    """Executing evaluation on a subset of properties:

    Args:
        alpha_class (float): Parameter used to calculate the class embedding.
        depth_class (int): Parameter used for the contextual depth of the class.
        depth_property (int): Parameter used for the contextual depth of the property.
        alpha_prop (float): Parameter used to calculate the property embedding.
        beta (float): Parameter used to calculate the property's associated classes embedding.
        gamma (float): Parameter used to calculate the property's types embedding.
        pooling_class (str): The applied pooling when combining multiple class embeddings.
        pooling_property (str): The applied pooling when combining multiple property embeddings.
        embedded_properties (dict): A dictionary of the properties in the ontology and their contextual embeddings.
        embedded_classes (dict): A dictionary of the classes in the ontology and their contextual embeddings.
        subset_properties (list): The properties, which will be evaluated/aligned.
        model_name (str): The name of the contextual model used to generate the embeddings.
        k_nearest (int): The nearest/most similar properties we will be looking for.

    Returns:
        None

    """
    embeddings_dict = {}
    for prop_name, curr_property in embedded_properties.items():
        enhanced_property_embedding = generate_enhanced_property_embedding(
            embedded_properties, embedded_classes, curr_property,
            depth_property, depth_class, depth_property,
            pooling_property, pooling_class,
            alpha_prop, alpha_class, beta, gamma,
        )
        embeddings_dict[prop_name] = enhanced_property_embedding

    subset_embeddings = np.array([embeddings_dict[prop] for prop in subset_properties])
    all_property_embeddings = np.array(list(embeddings_dict.values()))
    similarity_matrix = cosine_similarity(subset_embeddings, all_property_embeddings)

    # For each property in subset, find k-nearest neighbors and log results
    nearest_neighbors = {}
    property_class_associations = {}

    for i, prop in enumerate(subset_properties):
        # Find k-nearest neighbors
        similarities = similarity_matrix[i]

        # Sort indices based on similarities and limit to available indices
        nearest_indices = similarities.argsort()[-(k_nearest + 1):-1][::-1]

        # Ensure `nearest_indices` references `all_property_embeddings` not `subset_properties`
        nearest_props = [list(embeddings_dict.keys())[idx] for idx in nearest_indices]
        nearest_neighbors[prop] = nearest_props

        associated_classes = embedded_properties[prop].get("classes", [])
        property_class_associations[prop] = associated_classes

    print(nearest_neighbors)
    data = []
    for prop, neighbors in nearest_neighbors.items():
        curr_data = {
            "target_prop": prop,
            "target_classes": property_class_associations[prop],
            "mappings": []
        }
        for rank, neighbor in enumerate(neighbors, start=1):
            neighbor_classes = embedded_properties[neighbor].get("classes", [])
            for n_class in neighbor_classes:
                curr_data["mappings"].append(f"{neighbor} ({n_class})")

        data.append(curr_data)

    os.makedirs('./fine_tuning', exist_ok=True)
    with open(f'fine_tuning/data_{model_name}.json', 'w') as file:
        json.dump(data, file, indent=4)

    G = nx.Graph()

    # Add nodes and edges for each property and its neighbors
    for prop, neighbors in nearest_neighbors.items():
        # Add the main property node
        G.add_node(prop, label=prop, classes=property_class_associations[prop])

        # Add each neighbor with ranking and an edge between the property and its neighbor
        for rank, neighbor in enumerate(neighbors, start=1):
            # Find classes associated with the neighbor
            neighbor_classes = embedded_properties[neighbor].get("classes", [])
            G.add_node(neighbor, label=neighbor, classes=neighbor_classes)
            # Add edge with ranking info
            G.add_edge(prop, neighbor, rank=rank)

    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # Display the rank in the hover text
        edge_text.append(f"{G.nodes[edge[0]]['label']} - {G.nodes[edge[1]]['label']}<br>Rank: {edge[2]['rank']}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')

    # Create Plotly scatter plot for nodes
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
                        title='Property and Nearest Neighbor Relationships with Ranking',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',  # Set the plot background color to white
                        paper_bgcolor='white'
                    )
                    )
    fig.show()


def embed_data(classes: dict[str, dict], properties: dict[str, dict], model_name: str) -> tuple[dict, dict]:
    """Generate contextual embeddings of the properties and classes based on name and description.

    Args:
        classes (dict): A dictionary containing the classes in our ontology.
        properties (dict): A dictionary containing the properties in our ontology.
        model_name (str): The contextual model we would like to use to generate the embeddings.

    Returns:
        tuple: the embedded classes and properties.

    """
    embedded_classes, embedded_properties = {}, {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base').to(device)
    elif model_name == "deberta":
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        model = DebertaModel.from_pretrained('microsoft/deberta-base').to(device)
    else:
        tokenizer = None
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Preprocess classes
    for class_name, class_info in classes.items():
        if "description" not in class_info:
            synsets = get_synsets(class_info["title"])
            class_info["description"] = f"A class called '{class_info['title']}'. "
            if synsets:
                class_info["description"] = class_info["description"] + synsets[0].definition()

        embedded_classes[class_name] = {}
        embedded_classes[class_name]["embedding"] = generate_embedding(
            f"The class with title '{class_name}' has the following description: \"{class_info['description']}\"",
            model, tokenizer, device,
        )

        for parent_class in class_info.get("subClassOf", []):
            if "owl#" in parent_class:
                parent_class = parent_class.split("#")[1]
            if parent_class in classes:
                if "subClassOf" not in embedded_classes[class_name]:
                    embedded_classes[class_name]["subClassOf"] = []
                embedded_classes[class_name]["subClassOf"].append(parent_class)
            else:
                logging.warning(f"Domain class '{parent_class}' from class '{class_name}' not a class.")

    # Preprocess properties
    for prop_name, prop_info in properties.items():
        if "description" not in prop_info:
            synsets = get_synsets(prop_info["title"])
            prop_info[
                "description"] = f"A property called '{prop_info['title']}' which is part of the classes '{''.join(prop_info.get('classes', []))}'. "
            if synsets:
                prop_info["description"] = prop_info["description"] + synsets[0].definition()

        embedded_properties[prop_name] = {}
        embedded_properties[prop_name]["embedding"] = generate_embedding(
            f"The property with title '{prop_name}' has the following description: \"{prop_info['description']}\"",
            model, tokenizer, device,
        )

        for domain_class in prop_info.get("classes", []):
            if "owl#" in domain_class:
                domain_class = domain_class.split("#")[1]
            if domain_class in classes or domain_class in properties:
                if "classes" not in embedded_properties[prop_name]:
                    embedded_properties[prop_name]["classes"] = []
                embedded_properties[prop_name]["classes"].append(domain_class)
            else:
                logging.warning(f"Domain class '{domain_class}' from property '{prop_name}' not a class or property.")

        for parent_property in prop_info.get("subPropertyOf", []):
            if "owl#" in parent_property:
                parent_property = parent_property.split("#")[1]
            if parent_property in properties or parent_property in classes:
                if "subPropertyOf" not in embedded_properties[prop_name]:
                    embedded_properties[prop_name]["subPropertyOf"] = []
                embedded_properties[prop_name]["subPropertyOf"].append(parent_property)
            else:
                logging.warning(f"Property '{parent_property}' from property '{prop_name}' not a property or class.")

        for type_property in prop_info.get("types", []):
            if "owl#" in type_property:
                type_property = type_property.split("#")[1]
            if type_property in properties or type_property in classes:
                if "types" not in embedded_properties[prop_name]:
                    embedded_properties[prop_name]["types"] = []
                embedded_properties[prop_name]["types"].append(type_property)
            else:
                logging.warning(f"Type property '{type_property}' from property '{prop_name}' not a property or class.")

    return embedded_classes, embedded_properties


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

    program_type = "fine-tune"
    model_name = "roberta"  # bert, roberta, deberta, sbert
    embedded_classes, embedded_properties = embed_data(classes, properties, model_name)

    if program_type == "fine-tune":
        forbidden_classes = [
            "Car", "Vehicle", "BusOrCoach", "EngineSpecification", "MeanOfTransportation",
            "Order", "Offer", "Organization", "DrugCost", "ShippingDeliveryTime",
            "MedicalOrganization", "ProgramMembership", "Patient", "Legislation",
            "GovernmentService", "Drug", "MedicalEntity", "TherapeuticProcedure",
            "DrugStrength", "IndividualProduct", "ChemicalSubstance", "Substance"
        ]
        subset_properties = []
        for emb_name, emb_data in embedded_properties.items():
            if "classes" in emb_data:
                good_classes = set(emb_data["classes"]) - set(forbidden_classes)
                if good_classes:
                    subset_properties.append(emb_name)

        random.shuffle(subset_properties)
        subset_properties = subset_properties[:30]

        execute_evaluation(3, 3, 0.7,0.4, 0.2, 0.2, "mean", "mean", embedded_properties, embedded_classes, subset_properties, model_name)
    else:
        alpha_class_list = [0.5, 0.7, 0.9]  # Alpha for classes
        depths_list = [1, 2, 3]
        alpha_beta_gamma_combinations = [
            (0.4, 0.3, 0.2),  # Sum = 0.9, type_weight = 0.1
            (0.5, 0.2, 0.2),  # Sum = 0.9, type_weight = 0.1
            (0.4, 0.1, 0.4),  # Sum = 0.9, type_weight = 0.1
            (0.3, 0.3, 0.3),  # Sum = 0.9, type_weight = 0.1
            (0.7, 0.05, 0.2),  # Sum = 0.95, type_weight = 0.05
            (0.6, 0.25, 0.1),  # Sum = 0.95, type_weight = 0.05
            (0.35, 0.15, 0.45),  # Sum = 0.95, type_weight = 0.05
        ]
        pooling_methods = ['mean', 'sum', 'max']  # General pooling options
        n_clusters_list = [3, 5, 10, 20, 30, 40, 50, 75, 100]

        combinations = list(product(
            alpha_class_list,
            depths_list,
            alpha_beta_gamma_combinations,
            pooling_methods,
            pooling_methods,
        ))
        results = []
        clusters_data = {"labels": list(properties.keys())}
        subset_properties = [
            "payload",
            "roofLoad",
            "weightTotal",
            "employee",
            "product",
            "maximumIntake",
            "activeIngredient"
        ]

        for alpha_class, depth, (alpha_prop, beta, gamma), pooling_class, pooling_property in tqdm(combinations, desc="Best-param search"):
            # Generate combined embeddings for properties
            embeddings = []
            for prop_name, curr_property in embedded_properties.items():
                combined_embedding = generate_enhanced_property_embedding(
                    embedded_properties, embedded_classes, curr_property,
                    depth, depth, depth,
                    pooling_property, pooling_class,
                    alpha_prop, alpha_class, beta, gamma,
                )
                embeddings.append(combined_embedding)

            embeddings = np.array(embeddings)
            for n_clusters in n_clusters_list:
                kmeans = KMeans(n_clusters=n_clusters)
                labels = kmeans.fit_predict(embeddings)

                # Evaluate with clustering metrics
                silhouette_avg = silhouette_score(embeddings, labels)
                calinski_harabasz_avg = calinski_harabasz_score(embeddings, labels)
                davies_bouldin_avg = davies_bouldin_score(embeddings, labels)
                inertia = kmeans.inertia_

                # Store results for comparison
                results.append({
                    'alpha_class': float(alpha_class),
                    'alpha_prop': float(alpha_prop),
                    'beta': float(beta),
                    'gamma': float(gamma),
                    'depth': int(depth),
                    'pooling_class': pooling_class,
                    'pooling_property': pooling_property,
                    'n_clusters': int(n_clusters),
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz_avg),
                    'davies_bouldin_score': float(davies_bouldin_avg),
                    'inertia': float(inertia),
                })

                clusters_data[f'{alpha_class}_{alpha_prop}_{beta}_{gamma}_{depth}_{pooling_class}_{pooling_property}_{n_clusters}'] = labels.tolist()

        os.makedirs('./experiments', exist_ok=True)
        with open(f'./experiments/{model_name}_results.json', 'w') as f:
            json.dump(results, f)

        with open(f'./experiments/{model_name}_clusters.json', 'w') as f:
            json.dump(clusters_data, f)