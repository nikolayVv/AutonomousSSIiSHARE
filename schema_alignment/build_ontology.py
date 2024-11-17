import nltk
import logging

from utils.helper_functions import open_dataset
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

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

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"building_ontology_logging.log"),
            logging.StreamHandler(),
        ],
    )

    g = Graph()
    EX = Namespace("http://merged_ontology.org/ontology/")
    g.bind("ex", EX)
    data_dir = "./data/merged"
    classes, properties = open_dataset(data_dir)

    for class_name, class_info in classes.items():
        class_uri = EX[class_name]
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(class_info["title"])))

        if "description" not in class_info:
            synsets = get_synsets(class_info["title"])
            class_info["description"] = f"A class called '{class_info['title']}'. "
            if synsets:
                class_info["description"] = class_info["description"] + synsets[0].definition()
        g.add((class_uri, RDFS.comment, Literal(class_info["description"])))

        for parent_class in class_info.get("subClassOf", []):
            if "owl#" in parent_class:
                parent_class = parent_class.split("#")[1]
            if parent_class in classes:
                parent_class_uri = EX[parent_class]
                g.add((class_uri, RDFS.subClassOf, parent_class_uri))
            else:
                logging.warning(f"Domain class '{parent_class}' from class '{class_name}' not a class.")

    for prop_name, prop_info in properties.items():
        prop_uri = EX[prop_name]
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))  # or OWL.DatatypeProperty if applicable
        g.add((prop_uri, RDFS.label, Literal(prop_info["title"])))

        if "description" not in prop_info:
            synsets = get_synsets(prop_info["title"])
            prop_info["description"] = f"A property called '{prop_info['title']}' which is part of the classes '{''.join(prop_info.get('classes', []))}'. "
            if synsets:
                prop_info["description"] = prop_info["description"] + synsets[0].definition()
        g.add((prop_uri, RDFS.comment, Literal(prop_info["description"])))

        for domain_class in prop_info.get("classes", []):
            if domain_class in classes:
                domain_class_uri = EX[domain_class]
                g.add((prop_uri, RDFS.domain, domain_class_uri))
            else:
                logging.warning(f"Domain class '{domain_class}' from property '{prop_name}' not a class.")

        for parent_property in prop_info.get("subPropertyOf", []):
            if "DUL.owl" in parent_property:
                parent_property = parent_property.split("#")[1]
            if parent_property in properties:
                parent_property_uri = EX[parent_property]
                g.add((prop_uri, RDFS.subPropertyOf, parent_property_uri))
            else:
                logging.warning(f"Property '{parent_property}' from property '{prop_name}' not a property.")

    g.serialize(f"{data_dir}/merged_ontology.owl", format="xml")