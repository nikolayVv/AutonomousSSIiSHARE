import os
import json
import pprint
from openai import OpenAI
import pandas as pd
import logging

from utils.helper_functions import open_dataset
from tqdm import tqdm


def generate_data_prompt(dataset: list, data_type: str) -> str:
    """Generate the data part of the prompt.

    Args:
        dataset (list): list of the data that should be added to the prompt.
        data_type (str): the type of the data.

    Returns:
        str: the generated data part of the prompt.

    """
    data_prompt = ""
    counter = 1
    for data in dataset:
        if "title" in data:
            data_prompt += f"{f'{counter}. ' if enumerating else ''}{data_type}'s name: {data['title']}\n"
        if "description" in data:
            data_prompt += f"{f'{counter}. ' if enumerating else ''}{data_type}'s description: {data['description']}\n"
        if not disable_class and "classes" in data:
            data_prompt += f"{f'{counter}. ' if enumerating else ''}{data_type}'s corresponding classes: {', '.join(data['classes'])}\n"
        if "subPropertyOf" in data:
            data_prompt += f"{f'{counter}. ' if enumerating else ''}{data_type}'s parent properties: {', '.join(data['subPropertyOf'])}\n"
        if "subClassOf" in data:
            data_prompt += f"{f'{counter}. ' if enumerating else ''}{data_type}'s parent classes: {', '.join(data['subClassOf'])}\n"
        counter += 1

    return data_prompt.rstrip("\n")


def extract_additional_properties(properties, classes, curr_property, depth) -> tuple [set, set]:
    """Extracting additional properties data based on the depth.

    Args:
        properties (dict): A dictionary containing the properties in our ontology.
        classes (dict): A dictionary containing the classes in our ontology.
        curr_property (dict): The current property for which additional properties will be extracted.
        depth (int): The contextual depth of the current property.

    Returns:
        tuple: The additional properties and classes.

    """
    if depth == 1:
        return set(), set()

    additional_properties = set()
    additional_classes = set()

    for property_class in curr_property.get("classes", []):
        if property_class in classes:
            additional_classes.add(property_class)
            additional_classes.update(extract_additional_classes(classes, classes[property_class], depth - 1))

    for parent_property in curr_property.get("subPropertyOf", []):
        if parent_property in properties:
            additional_properties.add(parent_property)
            add_properties, add_classes = extract_additional_properties(properties, classes, properties[parent_property], depth - 1)
            additional_properties.update(add_properties)
            additional_classes.update(add_classes)

    return additional_properties, additional_classes


def extract_additional_classes(classes, curr_class, depth):
    if depth == 1:
        return set()

    additional_classes = set()
    for parent_class in curr_class.get("subClassOf", []):
        if parent_class in classes:
            additional_classes.add(parent_class)
            additional_classes.update(extract_additional_classes(classes, classes[parent_class], depth - 1))

    return additional_classes


def call_gpt_api(prompt_dict, client) -> str:
    """Call the GPT model with specific prompt.

    Args:
        prompt_dict (dict): The prompt that will be sent to the model.
        client (OpenAI): The client that will be used to call the model.

    Returns:
        str: Response of the model.

    """
    try:
        messages = [
            {"role": "system", "content": prompt_dict["system"]},
            {"role": "user", "content": prompt_dict["user"]}
        ]

        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:ishare:AQPypjCF", #"gpt-4o-mini"
            messages=messages
        )
        return response.choices[0].message.content # Return the assistant's response
    except Exception as e:
        print(f"API call failed: {e}")
        return ""


def generate_prompt(classes: dict, properties: dict, source_data: list[tuple[str, str]], target_data: list[str], depth: int, mappings: dict, threshold: float) -> dict[str, str]:
    """
    Generate a prompt for schema mapping tasks based on the model being used.

    Args:
        classes (dict): A dictionary containing the classes in our ontology.
        properties (dict): A dictionary containing the properties in our ontology.
        source_data (list): Source properties and their corresponding classes.
        target_data (list): Target classes.
        depth (int): The contextual depth that will be used when adding additional information.
        mappings (dict): Valid mappings used for fine-tuning prompt generation.
        threshold (float): The threshold of confidence that will be used in the prompt.

    Returns:
        dict: the generated prompt for the model.

    """
    base_instructions = (
        "You are an expert schema alignment AI tasked with mapping properties from one or more source class to one or more properties from one or more target classes based on similarity. Consider the following:\n\n"
        "- *Semantic Similarity*: Compare names and descriptions for related meanings.\n"
        "- *Contextual Similarity*: Consider how properties are used in their classes and their broader context.\n"
        "- *Parent Relations*: Account for similarities between parent classes and properties for hierarchical alignment.\n"
        "- *Multi-Attribute Mapping*: A source property may map to one or more target properties and vice versa if relevant.\n\n"
        "I will provide the source properties with descriptions, classes, and related attributes. Then, I will provide the target properties and schemas in a similar way. Parent properties/classes information may also be included for broader context."
    )

    source_class_names, source_property_names = set(), set()
    target_class_names, target_property_names = set(), set()
    additional_class_names, additional_property_names = set(), set()
    additional_classes, additional_properties = [], []
    source_classes, source_properties = [], []
    target_classes, target_properties = [], []

    for target_class in target_data:
        target_class_names.add(target_class)
        additional_class_names.update(extract_additional_classes(classes, classes[target_class], depth))

    for property_name, property_schema in properties.items():
        for target_class in target_data:
            if target_class in property_schema["classes"]:
                property_schema["classes"] = [target_class]
                target_property_names.add(property_name)

                add_properties, add_classes = extract_additional_properties(properties, classes, property_schema, depth)
                additional_property_names.update(add_properties)
                additional_class_names.update(add_classes)
        for source_property, source_class in source_data:
            if property_name == source_property:
                property_schema["classes"] = [source_class]
                source_property_names.add(property_name)
                source_class_names.add(source_class)

                add_properties, add_classes = extract_additional_properties(properties, classes, property_schema, depth)
                additional_property_names.update(add_properties)
                additional_class_names.update(add_classes)
                additional_class_names.update(extract_additional_classes(classes, classes[source_class], depth))

    additional_class_names = list(additional_class_names)
    additional_property_names = list(additional_property_names)

    for source_class_name in source_class_names:
        source_classes.append(classes[source_class_name])

    for source_property_name in source_property_names:
        source_properties.append(properties[source_property_name])

    for target_class_name in target_class_names:
        target_classes.append(classes[target_class_name])

    for target_property_name in target_property_names:
        target_properties.append(properties[target_property_name])

    for additional_class_name in additional_class_names:
        additional_classes.append(classes[additional_class_name])

    for additional_property_name in additional_property_names:
        additional_properties.append(properties[additional_property_name])

    source_information = (
        f"Map the following {len(source_properties)} source properties:\n"
        f"{generate_data_prompt(list(source_properties), 'Source property')}\n"
        f"The source properties are part of the following {len(source_classes)} classes:\n"
        f"{generate_data_prompt(list(source_classes), 'Source class')}"
    )

    target_information = (
        f"Map the source properties to one or more of these {len(target_properties)} target properties:\n"
        f"{generate_data_prompt(list(target_properties), 'Target property')}\n"
        f"The target properties are part of the following {len(target_classes)} classes:\n"
        f"{generate_data_prompt(list(target_classes), 'Target class')}"
    )

    additional_information = None
    if len(additional_properties) > 0 and len(additional_classes) > 0:
        additional_information = (
            f"Additional {len(additional_properties)} context properties (for reference only):\n"
            f"{generate_data_prompt(additional_properties, 'Additional property')}\n"
            f"Additional {len(additional_classes)} context classes (for reference only):\n"
            f"{generate_data_prompt(additional_classes, 'Additional class')}"
        )
    elif len(additional_properties) > 0 and len(additional_classes) == 0:
        additional_information = (
            f"Additional properties for reference only:\n"
            f"{generate_data_prompt(additional_properties, 'Additional property')}"
        )
    elif len(additional_properties) == 0 and len(additional_classes) > 0:
        additional_information = (
            f"Additional classes for reference only:\n"
            f"{generate_data_prompt(additional_classes, 'Additional class')}"
        )

    task_description = (
        "Your task is to map each source property to valid target properties to solve a critical problem. Let's work this out step by step to ensure accuracy.\n"
        f"Return a JSON that contains the source properties as keys and lists of mapping strings as the values. Each mapping string should follow the format 'target_property_1 (corresponding_class), ..., target_property_N (corresponding_class)', where none of the target properties should be the same as the source properties, otherwise the corresponding mapping is not valid. The mappings should be also ordered by confidence, highest to lowest. If for some key (source property) no mappings are valid and/or none of them have at least {threshold*100}% confidence, return an empty list ([]) as the corresponding value."
    )

    if additional_information is not None:
        user_content = f"{source_information}\n\n{target_information}\n\n{additional_information}\n\n{task_description}"
    else:
        user_content = f"{source_information}\n\n{target_information}\n\n{task_description}"

    if len(mappings):
        prompt = {
            "system": base_instructions,
            "user": user_content,
            "assistant": str(mappings),
        }
    else:
        prompt = {
            "system": base_instructions,
            "user": user_content,
        }

    return prompt


def evaluate_prompts(classes: dict[str, dict], properties: dict[str, dict], type: str, client: OpenAI) -> None:
    """Evaluate different types of prompts.

    Args:
        classes (dict): A dictionary containing the classes in our ontology.
        properties (dict): A dictionary containing the properties in our ontology.
        type (str): The type of the prompt that will be generated, i.e. number of source and target components.
        client (OpenAI): The client that will be used to call the model.

    Returns:
        None

    """
    if type == "1:1":
        data = [
            {
                "source_attributes": [("roofLoad", "Car")],
                "target_schemas": [["Vehicle"], ["Car"], ["BusOrCoach"], ["EngineSpecification"], ["MeanOfTransportation"]]
            },
            {
                "source_attributes": [("seller", "Offer")],
                "target_schemas": [["Order"], ["Offer"], ["Organization"], ["DrugCost"], ["ShippingDeliveryTime"]]
            },
            {
                "source_attributes": [("organizer", "Organization")],
                "target_schemas": [["Organization"], ["MedicalOrganization"], ["ProgramMembership"], ["Patient"]]
            },
            {
                "source_attributes": [("jurisdiction", "Legislation")],
                "target_schemas": [["Organization"], ["Legislation"], ["GovernmentService"], ["ProgramMembership"]]
            },
            {
                "source_attributes": [("activeIngredient", "Drug")],
                "target_schemas": [["Drug"], ["MedicalEntity"], ["TherapeuticProcedure"], ["DrugStrength"], ["IndividualProduct"], ["ChemicalSubstance"]]
            },
        ]
    elif type == "1:M":
        data = [
            {
                "source_attributes": [("roofLoad", "Car")],
                "target_schemas": [
                    ["Vehicle", "Car"], ["Vehicle", "BusOrCoach"], ["Vehicle", "EngineSpecification"], ["Vehicle", "MeanOfTransportation"],
                    ["Vehicle", "Car", "BusOrCoach"], ["Vehicle", "EngineSpecification", "MeanOfTransportation"],
                    ["Vehicle", "Car", "BusOrCoach", "EngineSpecification", "MeanOfTransportation"]
                ]
            },
            {
                "source_attributes": [("seller", "Offer")],
                "target_schemas": [
                    ["Order", "Offer"], ["Order", "Organization"], ["Order", "DrugCost"], ["Order", "ShippingDeliveryTime"],
                    ["Order", "Offer", "Organization"], ["Order", "DrugCost", "ShippingDeliveryTime"],
                    ["Order", "Offer", "Organization", "DrugCost", "ShippingDeliveryTime"]
                ]
            },
            {
                "source_attributes": [("organizer", "Organization")],
                "target_schemas": [
                    ["Organization", "MedicalOrganization"], ["Organization", "ProgramMembership"], ["Organization", "Patient"],
                    ["Organization", "MedicalOrganization", "ProgramMembership"], ["Organization", "ProgramMembership", "Patient"],
                    ["Organization", "MedicalOrganization", "ProgramMembership", "Patient"]
                ]
            },
            {
                "source_attributes": [("jurisdiction", "Legislation")],
                "target_schemas": [
                    ["Organization", "Legislation"], ["Organization", "GovernmentService"], ["Organization", "ProgramMembership"],
                    ["Organization", "Legislation", "GovernmentService"], ["Organization", "GovernmentService", "ProgramMembership"],
                    ["Organization", "Legislation", "GovernmentService", "ProgramMembership"],
                ]
            },
            {
                "source_attributes": [("activeIngredient", "Drug")],
                "target_schemas": [
                    ["Drug", "MedicalEntity"], ["Drug", "TherapeuticProcedure"], ["Drug", "DrugStrength"], ["Drug", "IndividualProduct"], ["Drug", "ChemicalSubstance"],
                    ["Drug", "MedicalEntity", "TherapeuticProcedure"], ["DrugStrength", "IndividualProduct", "ChemicalSubstance"],
                    ["Drug", "MedicalEntity", "TherapeuticProcedure", "DrugStrength", "IndividualProduct", "ChemicalSubstance"],
                ]
            },
        ]
    elif type == "N:1":
        data = [
            {
                "source_attributes": [("roofLoad", "Car"), ("payload", "Vehicle"), ("weightTotal", "Vehicle")],
                "target_schemas": [["Vehicle"], ["Car"], ["BusOrCoach"], ["EngineSpecification"], ["MeanOfTransportation"]]
            },
            {
                "source_attributes": [("seller", "Offer"), ("seller", "Order")],
                "target_schemas": [["Order"], ["Offer"], ["Organization"], ["DrugCost"], ["ShippingDeliveryTime"]]
            },
            {
                "source_attributes": [("organizer", "Organization"), ("medicalSpecialty", "MedicalOrganization"), ("jurisdiction", "Legislation")],
                "target_schemas": [["Organization"], ["MedicalOrganization"], ["ProgramMembership"], ["Patient"], ["Legislation"], ["GovernmentService"]]
            },
            {
                "source_attributes": [("activeIngredient", "Drug"), ("maximumIntake", "Substance")],
                "target_schemas": [["Drug"], ["MedicalEntity"], ["TherapeuticProcedure"], ["DrugStrength"], ["IndividualProduct"], ["ChemicalSubstance"]]
            },
        ]
    else:
        data = [
            {
                "source_attributes": [("roofLoad", "Car"), ("payload", "Vehicle"), ("weightTotal", "Vehicle")],
                "target_schemas": [
                    ["Vehicle", "Car"], ["Vehicle", "BusOrCoach"], ["Vehicle", "EngineSpecification"],
                    ["Vehicle", "MeanOfTransportation"],
                    ["Vehicle", "Car", "BusOrCoach"], ["Vehicle", "EngineSpecification", "MeanOfTransportation"],
                    ["Vehicle", "Car", "BusOrCoach", "EngineSpecification", "MeanOfTransportation"]
                ]
            },
            {
                "source_attributes": [("seller", "Offer"), ("seller", "Order")],
                "target_schemas": [
                    ["Order", "Offer"], ["Order", "Organization"], ["Order", "DrugCost"],
                    ["Order", "ShippingDeliveryTime"],
                    ["Order", "Offer", "Organization"], ["Order", "DrugCost", "ShippingDeliveryTime"],
                    ["Order", "Offer", "Organization", "DrugCost", "ShippingDeliveryTime"]
                ]
            },
            {
                "source_attributes": [("organizer", "Organization"), ("medicalSpecialty", "MedicalOrganization"), ("jurisdiction", "Legislation")],
                "target_schemas": [
                    ["Organization", "MedicalOrganization"], ["Organization", "ProgramMembership"],
                    ["Organization", "Patient"], ["Organization", "Legislation"], ["Organization", "GovernmentService"],
                    ["Organization", "MedicalOrganization", "ProgramMembership"], ["Organization", "Legislation", "GovernmentService"],
                    ["Organization", "ProgramMembership", "Patient"], ["Organization", "GovernmentService", "ProgramMembership"],
                    ["Organization", "MedicalOrganization", "ProgramMembership", "Patient", "Legislation", "GovernmentService"]
                ]
            },
            {
                "source_attributes": [("activeIngredient", "Drug"), ("maximumIntake", "Substance")],
                "target_schemas": [
                    ["Drug", "MedicalEntity"], ["Drug", "TherapeuticProcedure"], ["Drug", "DrugStrength"],
                    ["Drug", "IndividualProduct"], ["Drug", "ChemicalSubstance"],
                    ["Drug", "MedicalEntity", "TherapeuticProcedure"],
                    ["DrugStrength", "IndividualProduct", "ChemicalSubstance"],
                    ["Drug", "MedicalEntity", "TherapeuticProcedure", "DrugStrength", "IndividualProduct",
                     "ChemicalSubstance"],
                ]
            },
        ]

    results = []
    for data_object in tqdm(data, desc=f"Evaluating LLM type {type}"):
        source_attributes = data_object["source_attributes"]
        for target_schemas in data_object["target_schemas"]:
            for depth in [1, 3, 5]:
                for threshold in [0.5, 0.7, 0.9]:
                    for execution in range(10):
                        prompt = generate_prompt(classes, properties, source_attributes, target_schemas, depth, [],
                                                 threshold)
                        response = call_gpt_api(prompt, client)

                        results.append({
                            'source_attributes': ", ".join([f"{source_attribute} ({source_schema})" for source_attribute, source_schema in source_attributes]),
                            'target_schemas': ", ".join(target_schemas),
                            'mapping_type': f"{len(source_attributes)}:{len(target_schemas)}",
                            'depth': depth,
                            'execution_number': execution,
                            'threshold': threshold,
                            'response': response
                        })

    df = pd.DataFrame(results)
    os.makedirs('./prompts', exist_ok=True)
    df.to_parquet(f'prompts/evaluation_results_{"_".join(type.split(":"))}.parquet', index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation.log"),
            logging.StreamHandler(),
        ],
    )
    program_type = "fine-tune"
    label = "test"
    mappings = {}
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    depth = 4
    data_path = f"./prompts/prompt_{label}_d{depth}.jsonl"
    classes, properties = open_dataset("./data/merged")

    if program_type == "fine-tune":
        with open('fine_tuning/mappings.json') as f:
            data = json.load(f)

        samples = []
        for data_object in data:
            prompt = generate_prompt(classes, properties, data_object["source_attributes"], data_object["target_schemas"], 3, data_object["mappings"], data_object["threshold"])
            samples.append({
                "messages": [
                {
                    "role": "system",
                    "content": prompt["system"],
                },
                {
                    "role": "user",
                    "content": prompt["user"],
                },
                {
                    "role": "assistant",
                    "content": prompt["assistant"],
                }
                ]
            })

        with open('fine_tuning/samples.jsonl', 'w') as file:
            for item in samples:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
    else:
        evaluate_prompts(classes, properties, "1:1", client)
        evaluate_prompts(classes, properties, "1:M", client)
        evaluate_prompts(classes, properties, "N:1", client)
        evaluate_prompts(classes, properties, "N:M", client)