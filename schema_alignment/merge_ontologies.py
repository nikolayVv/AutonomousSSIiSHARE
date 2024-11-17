import logging
import json
import os
import time

from fuzzywuzzy import fuzz
from tqdm import tqdm
from utils.helper_functions import open_dataset

def merge_classes(classes_dict, similarity_threshold=90)-> tuple[dict, list]:
    """Merge the classes.

    Args:
        classes_dict (dict): A dictionary of the classes in the ontology.
        similarity_threshold (int): Similarity threshold used for merging of descriptions.

    Returns:
        tuple: Dictionary of the merged classes and a list of the same classes.

    """
    merged_classes = {}
    same_classes = []

    for database, classes in classes_dict.items():
        for schema, data in tqdm(classes.items(), desc=f"Merging {database} Classes"):
            if schema not in merged_classes:
                merged_classes[schema] = data
            else:
                if schema not in same_classes:
                    same_classes.append(schema)
                if "description" in data:
                    if "description" in merged_classes[schema]:
                        similarity = fuzz.ratio(
                            merged_classes[schema]["description"],
                            data["description"]
                        )
                        if similarity < similarity_threshold:
                            logging.info(f"[{database}] Same class, but different descriptions with similarity: {similarity}")
                            logging.info(merged_classes[schema])
                            logging.info(data)
                            merged_classes[schema]['description'] += "\n" + data["description"]
                    else:
                        merged_classes[schema]["description"] = data["description"]

                if "subClassOf" in data:
                    if "subClassOf" not in merged_classes[schema]:
                        merged_classes[schema]["subClassOf"] = []

                    for subclass in data["subClassOf"]:
                        if subclass not in merged_classes[schema]["subClassOf"]:
                            logging.info(f"[{database}] Same class, but different subclassses")
                            logging.info(merged_classes[schema])
                            logging.info(data)
                            merged_classes[schema]["subClassOf"].append(subclass)

            if "ontology" not in merged_classes[schema]:
                merged_classes[schema]["ontology"] = []
            merged_classes[schema]["ontology"].append(database)

    return merged_classes, same_classes


def merge_properties(properties_dict, similarity_threshold=90) -> tuple[dict, list]:
    """Merge the properties

    Args:
        properties_dict (dict): A dictionary of the properties in the ontology.
        similarity_threshold (int): Similarity threshold used for merging of descriptions.

    Returns:
        tuple: Dictionary of the merged properties and a list of the same properties.

    """
    merged_properties = {}
    same_properties = []

    for database, classes in properties_dict.items():
        for property, data in tqdm(classes.items(), desc=f"Merging {database} Properties"):
            if "classes" in data:
                if property not in merged_properties:
                    merged_properties[property] = data
                else:
                    if property not in same_properties:
                        same_properties.append(property)
                    if "description" in data:
                        if "description" in merged_properties[property]:
                            similarity = fuzz.ratio(
                                merged_properties[property]["description"],
                                data["description"]
                            )
                            if similarity < similarity_threshold:
                                logging.info(f"[{database}] Same property, but different descriptions with similarity: {similarity}")
                                logging.info(merged_properties[property])
                                logging.info(data)
                                merged_properties[property]['description'] += "\n" + data["description"]
                        else:
                            merged_properties[property]["description"] = data["description"]

                    if "subPropertyOf" in data:
                        if "subPropertyOf" not in merged_properties[property]:
                            merged_properties[property]["subPropertyOf"] = []

                        for subproperty in data["subPropertyOf"]:
                            if subproperty not in merged_properties[property]["subPropertyOf"]:
                                logging.info(f"[{database}] Same class, but different subproperties")
                                logging.info(merged_properties[property])
                                logging.info(data)
                                merged_properties[property]["subPropertyOf"].append(subproperty)

                    if "classes" in data:
                        if "classes" not in merged_properties[property]:
                            merged_properties[property]["classes"] = []

                        for parent_class in data["classes"]:
                            if parent_class not in merged_properties[property]["classes"]:
                                logging.info(f"[{database}] Same property, but different classes")
                                logging.info(merged_properties[property])
                                logging.info(data)
                                merged_properties[property]["classes"].append(parent_class)

                    if "types" in data:
                        if "types" not in merged_properties[property]:
                            merged_properties[property]["types"] = []

                        for prop_type in data["types"]:
                            if prop_type not in merged_properties[property]["types"]:
                                logging.info(f"[{database}] Same property, but different types")
                                logging.info(merged_properties[property])
                                logging.info(data)
                                merged_properties[property]["types"].append(prop_type)

                if "ontology" not in merged_properties[property]:
                    merged_properties[property]["ontology"] = []
                merged_properties[property]["ontology"].append(database)

    return merged_properties, same_properties


def merge_data() -> tuple[dict, dict]:
    """Merge multiple dictionaries of classes or properties.

    Returns:
        tuple: The merged classes and properties.

    """
    schema_org_classes, schema_org_properties = open_dataset("./data/schema_org")
    dbpedia_classes, dbpedia_properties = open_dataset("./data/dbpedia")
    yago_classes, yago_properties = open_dataset("./data/yago")

    start = time.time()
    merged_classes, same_classes = merge_classes({
        "Schema ORG": schema_org_classes,
        "DBpedia": dbpedia_classes,
        "YAGO": yago_classes
    })

    merged_properties, same_properties = merge_properties({
        "Schema ORG": schema_org_properties,
        "DBpedia": dbpedia_properties,
        "YAGO": yago_properties
    })

    end = time.time()
    logging.info(f"Merged schemas for {end - start} seconds with {len(merged_classes)} classes and {len(merged_properties)} properties.")
    logging.info(f"Same schemas between the ontologies ({len(same_classes)}): {same_classes}.")
    logging.info(f"Same schemas between the ontologies ({len(same_properties)}): {same_properties}.")

    return merged_classes, merged_properties


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"data_merging.log"),
            logging.StreamHandler(),
        ],
    )

    data_dir = "./data/merged"

    classes, properties = merge_data()

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    with open(f'{data_dir}/classes.json', 'w') as f:
        json.dump(classes, f, indent=2)
    with open(f'{data_dir}/properties.json', 'w') as f:
        json.dump(properties, f, indent=2)