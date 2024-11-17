import rdflib
import logging
import time
import json
import os

from tqdm import tqdm


def preprocess_yago() -> tuple[dict, dict]:
    """Fetches and preprocesses all schemas from YAGO ontology.

    Returns:
        tuple: the preprocessed classes and properties in JSON format.

    """
    g = rdflib.Graph()
    g.parse("./data/yago/yago-raw.nt", format="nt")

    start = time.time()
    classes = {}
    properties = {}
    total_triples = sum(1 for _ in g)  # Calculate the total number of triples
    pbar = tqdm(total=total_triples, desc="Preprocessing YAGO Schemas", unit="triple")

    for s, p, o in g:
        schema_type, attribute_label = p.split("/")[-1].split("#")
        if schema_type == "rdf-schema":
            schema_label = s.split("/")[-1]
            if schema_label[0].isupper():
                if schema_label not in classes:
                    classes[schema_label] = { "title": schema_label }

                if attribute_label == "comment":
                    classes[schema_label]["description"] = o
                elif attribute_label == "subClassOf":
                    if attribute_label not in classes[schema_label]:
                        classes[schema_label][attribute_label] = []

                    classes[schema_label][attribute_label].append(o.split("/")[-1])
                elif attribute_label != "label":
                    print(attribute_label)
            else:
                if schema_label not in properties:
                    properties[schema_label] = { "title": schema_label }

                if attribute_label == "comment":
                    properties[schema_label]["description"] = o
                elif attribute_label == "subPropertyOf":
                    if attribute_label not in properties[schema_label]:
                        properties[schema_label][attribute_label] = []

                    properties[schema_label][attribute_label].append(o.split("/")[-1])
                elif attribute_label == "range":
                    if "types" not in properties[schema_label]:
                        properties[schema_label]["types"] = []

                    properties[schema_label]["types"].append(o.split("/")[-1])
                elif attribute_label == "domain":
                    if "classes" not in properties[schema_label]:
                        properties[schema_label]["classes"] = []

                    properties[schema_label]["classes"].append(o.split("/")[-1])
                elif attribute_label != "label":
                    print(attribute_label)

        pbar.update(1)  # Increment progress bar by 1 for each triple processed

    pbar.close()

    end = time.time()
    logging.info(f"Preprocessed schemas from YAGO for {end - start} seconds with {len(classes)} classes and {len(properties)} properties.")

    return classes, properties


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"data_preprocessing.log"),
            logging.StreamHandler(),
        ],
    )

    data_dir = "./data/yago"

    classes, properties = preprocess_yago()

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    with open(f'{data_dir}/classes.json', 'w') as f:
        json.dump(classes, f, indent=2)
    with open(f'{data_dir}/properties.json', 'w') as f:
        json.dump(properties, f, indent=2)