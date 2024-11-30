# Autonomous SSI-enhanced iSHARE Framework

This repository implements and evaluates an autonomous Self-Sovereign Identity (SSI)-enhanced iSHARE framework. By leveraging advanced schema alignment techniques and large language models (LLMs), this framework explores how decentralized identities and intelligent schema matching can optimize verification processes within data-sharing ecosystems. The implementation focuses on a specific use case from the associated research and is not intended for real-world systems. Instead, it provides a foundation for exploring the integration of SSI technologies into the iSHARE framework.

## Usage

The implementation in this repository is tailored to evaluate the integration of SSI technologies with the iSHARE framework. The focus is on optimizing the evaluation process by employing specific schemas, policies, and configurations. Some functionalities may require the addition of policies, credentials, and sample data to work as expected. These adjustments are necessary to replicate the controlled conditions described in the research.

This code demonstrates the potential of SSI technologies in data-sharing scenarios but does not follow production best practices. Instead, it showcases a practical and research-oriented setup to analyze the impact of these technologies.
### iSHARE Implementation

The repository includes three versions of the iSHARE framework, each demonstrating different levels of enhancement:

1. **iSHAREv1**: A classic, centralized iSHARE implementation and its currently available functionalities adapted for the specific use case.
2. **iSHAREv2**: An enhanced decentralized version of iSHAREv1 that introduces SSI technologies, such as decentralized identifiers (DIDs), verifiable credentials (VCs) and verifiable presentations (VPs).
3. **iSHAREv3**: An enhanced version of iSHAREv3 that leverages large language models (LLMs) for advanced schema alignment and automated alternative VP request generation in the SSI architecture.

All functionalities from all the three version are implemented and can be found in the `app_api` directory. The solution is implemented in the form of an agent, i.e. API consisting of multiple endpoints for different functionalities of the different versions.

#### Running Locally

The solution can be executed locally, by first installing all the needed commands:
```
npm install
```

After the dependencies are installed, a custom instance of the agent can be started by preparing the required environmental variables with the desired configuration. After the configuration for a specific instance is prepared it can be executed in its own terminal by writing the following command in the `app_api` directory:
```
npm run start
```

You can run multiple instances of the solution with different configurations simultaneously. These instances can interact with each other locally, allowing for a distributed setup and testing of cross-instance interactions.

#### Running on Docker

For convenience, the repository includes a `Dockerfile` and a `docker-compose.yml` script to simplify execution. These are pre-configured to generate the instances described in the research scenario. The solution can be run as Docker containers by simply writing the following command:
```
docker-compose up
```

Of course, the configurations for the Docker containers can be easily adapted to run different scenarios and instances. Update the docker-compose.yml file as needed to define custom setups. This allows you to replicate the use-case scenarios presented in the research and/or create your own instances for evaluation and experimentation.

### Schema Alignment Evaluation

The integration of LLMs enables advanced schema alignment to support property matching between different classes. This capability enhances interoperability and adaptability in data-sharing frameworks. Furthermore, the schema alignment process facilitates the automation of alternative Verifiable Presentation (VP) requests or Selective Disclosure Requests (SDRs) within SSI ecosystems.

The scripts in the `schema_alignment` directory can be used to recreate the experiments conducted in the research, but to also extend and improve the results by integrating additional techniques, models, or combinations of methods to explore new possibilities for schema alignment and verification workflows. The available scripts can be used to:
- preprocess `*_preprocess.py`, merge `merge_ontologies.py` and build `build_ontology.py` ontologies into the data format described in the research.
- generating graph `graph_similarity.py` and context `context_similarity.py` embeddings using different parameters and evaluating the generated embeddings.
- prompting and evaluating schema property mappings using LLM models `prompt_evaluation.py`.
- human evaluating generated property mappings based on the scores specified in the research.

## Contribution

This repository implements the components and services needed to achieve the system described in the research. It provides a starting point for further exploration and development in integrating LLM schema alignment and SSI technologies into data-sharing frameworks. Pull requests are open, and developers are encouraged to contribute. For changes, please fork the repository!

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

This solution is developed as part of my master's thesis at the [University of Ljubljana Faculty of Computer and Information Science](https://www.fri.uni-lj.si/en). I would like to extend my deepest gratitude to:
- My mentor, [Assoc. Prof. dr. Dejan Lavbič](https://www.lavbic.net/), who introduced me to the field of data spaces and provided invaluable guidance and support throughout the research and development of this project.
- The [Veramo](https://github.com/decentralized-identity/veramo) and [iSHARE](https://github.com/iSHAREScheme) communities for their existing contributions, which significantly optimized the research and development process.
