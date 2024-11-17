import requests
import logging
import time
import json
import os

from tqdm import tqdm


def preprocess_schema_org() -> tuple[dict, dict]:
    """Fetches and preprocesses all schemas from Schema.org ontology.

    Returns:
        tuple: the preprocessed classes and properties in JSON format.

    """
    response = requests.get("https://schema.org/version/latest/schemaorg-current-https.jsonld")

    if response.status_code != 200:
        logging.warning("Couldn't extract and preprocess Schema.org schemas")
        return {}, {}

    data = response.json()
    classes = {}
    start = time.time()

    # Extract classes
    for entity in tqdm(data['@graph'], desc="Preprocessing Schema.org Classes"):
        if entity.get('@type') == 'rdfs:Class':
            label = entity.get('rdfs:label')
            if type(label) is dict:
                label = label.get('@value')

            classes[label] = { "title": label }

            if "rdfs:comment" in entity:
                description = entity.get('rdfs:comment')
                if "@value" in description:
                    description = description["@value"]
                classes[label]["description"] = description

            if "rdfs:subClassOf" in entity:
                classes[label]["subClassOf"] = []
                parent_schemas = entity.get('rdfs:subClassOf')
                if type(parent_schemas) is not list:
                    parent_schemas = [parent_schemas]
                for parent_schema in parent_schemas:
                    classes[label]["subClassOf"].append(parent_schema.get('@id').split(':')[-1])

    # Add properties to classes
    properties = {}
    for entity in tqdm(data['@graph'], desc="Preprocessing Schema.org Properties"):
        if entity.get('@type') == 'rdf:Property':
            domains = entity.get('schema:domainIncludes')
            if domains is None:
                logging.warning(f"No domains for {entity}.")
            else:
                if type(domains) is not list:
                    domains = [domains]

                domains = [domain.get('@id').split(':')[-1] for domain in domains]
                property_label = entity.get('rdfs:label')
                if type(property_label) is dict:
                    property_label = property_label.get('@value')

                curr_property = { "title": property_label, "classes": domains}
                if "rdfs:comment" in entity:
                    description = entity.get('rdfs:comment')
                    if "@value" in description:
                        description = description["@value"]
                    curr_property["description"] = description

                if "schema:rangeIncludes" in entity:
                    curr_property["types"] = []
                    property_types = entity.get('schema:rangeIncludes')
                    if type(property_types) is not list:
                        property_types = [property_types]
                    for property_type in property_types:
                        curr_property["types"].append(property_type.get('@id').split(':')[-1])

                if "rdfs:subPropertyOf" in entity:
                    curr_property["subPropertyOf"] = []
                    parent_properties = entity.get('rdfs:subPropertyOf')
                    if type(parent_properties) is not list:
                        parent_properties = [parent_properties]
                    for parent_property in parent_properties:
                        curr_property["subPropertyOf"].append(parent_property.get('@id').split(':')[-1])

                properties[property_label] = curr_property

    end = time.time()
    logging.info(f"Preprocessed schemas from Schema.org for {end - start} seconds with {len(classes)} classes and {len(properties)} properties.")

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
    data_dir = "./data/schema_org"

    classes, properties = preprocess_schema_org()

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    with open(f'{data_dir}/classes.json', 'w') as f:
        json.dump(classes, f, indent=2)
    with open(f'{data_dir}/properties.json', 'w') as f:
        json.dump(properties, f, indent=2)