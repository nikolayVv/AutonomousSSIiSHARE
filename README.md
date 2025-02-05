# Autonomous SSI-enhanced iSHARE Framework

This repository implements and evaluates an autonomous Self-Sovereign Identity (SSI)-enhanced iSHARE framework. By leveraging advanced schema alignment techniques and large language models (LLMs), this framework explores how decentralized identities and intelligent schema matching can optimize verification processes within data-sharing ecosystems. The implementation focuses on a specific use case from the associated research and is not intended for real-world systems. Instead, it provides a foundation for exploring the integration of SSI technologies into the iSHARE framework.

## Usage

The implementation in this repository is tailored to evaluate the integration of SSI technologies with the iSHARE framework. The focus is on optimizing the evaluation process by employing specific schemas, policies, and configurations. Some functionalities may require the addition of policies, credentials, and sample data to work as expected. These adjustments are necessary to replicate the controlled conditions described in the research.

This code demonstrates the potential of SSI technologies in data-sharing scenarios but does not follow production best practices. Instead, it showcases a practical and research-oriented setup to analyze the impact of these technologies.

### iSHARE Implementation

The repository includes three versions of the iSHARE framework, each demonstrating different levels of enhancement:

1. **iSHAREv1**: A classic, centralized iSHARE implementation and its currently available functionalities adapted for the specific use case.
2. **iSHAREv2**: An enhanced decentralized version of iSHAREv1 that introduces SSI technologies, such as decentralized identifiers (DIDs), verifiable credentials (VCs) and verifiable presentations (VPs).
3. **iSHAREv3**: An enhanced version of iSHAREv3 that leverages large language models (LLMs) for advanced schema alignment and automated alternative VP request (SDR) generation in the SSI architecture.

All functionalities from all the three version are implemented and can be found in the `app_api` directory. The solution is implemented in the form of an agent, i.e. API consisting of multiple endpoints for different functionalities of the different versions.

#### Results
![image (46)](https://github.com/user-attachments/assets/56e4450a-e6b3-4b67-8a16-293807f389b8)

To evaluate the implementations, we conducted a case study that assesses how each version of the iSHARE framework handles complex interactions between entities. The case study includes two data spaces, allowing us to test interactions between entities within the same satellite (inter-satellite) and across two different satellites (intra-satellite). Each satellite in our case study represents the main control entity of a data space. Additionally, we created both positive and negative versions of the scenarios, focusing on key metrics such as scenario duration (in number of API calls/requests), scalability, flexibility, and error handling.

| Scenario/Evaluation Metric   | iSHAREv1 | iSHAREv2  | iSHAREv3    |
| ---------------------------- | -------- | --------- | ----------- |
| **Positive Intra-Satellite** | **6**/20 | 12/**20** | 12/**20**   |
| **Negative Intra-Satellite** | -/-      | -/-       | **14+/22+** |
| **Positive Inter-Satellite** | **8**/32 | 16/**28** | 16/**28**   |
| **Negative Inter-Satellite** | -/-      | -/-       | **18+/30+** |
|                              |          |           |             |
| **Scalability**              | 2        | **5**     | 4           |
| **Flexibility**              | 2        | 4         | **5**       |
| **Error Handling**           | 2        | 3         | **5**       |
|                              |          |           |             |

The first thing visible in the table is the total execution steps across the scenarios, where each cell represents the following two initial states of the data spaces:
1. Left-hand side: The case where the entities interact with each other for the first time, i.e., no connections were previously established.
2. Right-hand side: The case where some entities have already interacted before the execution of the scenarios, i.e., some connections were previously established.
   
We can see that for the first (left) initial state, the original version of the framework performs best in both positive scenarios due to its simpler architecture. However, in real-world scenarios, iSHAREv1 faces limitations due to repeated connection setups, which introduce significant overhead. In this initial state (right), the SSI-enhanced versions overcome these limitations by enabling entities to establish persistent, secure, and reusable connections. Moreover, iSHAREv3's ability to generate alternative SDRs (VP requests) allows it to handle failed verification scenarios with minimal additional steps.

The table also shows that the original iSHARE version, while straightforward in design, struggles with scalability and adaptability due to its reliance on central entities (Satellite, Authorization Registry), resulting in bottlenecks and a need for manual intervention in error handling. iSHAREv2’s decentralized model with DIDs, VCs, and VPs enables efficient scaling, handling a large number of requests seamlessly without degrading performance, while also offering greater adaptability to changing requirements. Its error handling is improved through decentralized trust but still requires occasional manual input for policy-related issues. iSHAREv3 takes the highest levels of flexibility and error handling automation, dynamically adapting to verification needs and fully automating the recovery process. Its AI-driven approach enables the generation of alternative VPs and the retrying of failed verification processes without manual intervention in the schema alignment process. However, retrying failed verifications and recovering from errors may cause connections to remain active longer than expected and may not always result in successful schema mappings, potentially impacting system performance.

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

#### Results

![Schema_Matching_Architecture (2)](https://github.com/user-attachments/assets/28096dd6-ae46-432a-b29b-f64488b0511a)

In iSHAREv3, entities with the role of verifier will be able to execute the alternative SDR generation in case of invalid verification. In this case, the verifier will be able to request all the necessary credential schemas to be used in the schema alignment process. The SSI Agent of the entity will then generate a specific prompt and send it to the LLM model, which is GPT-4o-Mini in our case. The model will return candidate mappings, ordered by confidence and above a certain threshold. Based on the provided mappings, the entity must then decide whether to:
1. Accept all the generated mappings as sufficient for the generation of the alternative SDR, or
2. Determine that some or all of the mappings are inadequate, requiring another execution of the schema alignment process with a more specific prompt, or
3. Decide that some or all of the mappings are inadequate and terminate the verification process.

The evaluation of schema alignment via Large Language Models (GPT-4o-Mini) was conducted through human evaluation, considering the relevance of mappings, order of mappings by confidence, completeness, and interpretability, for a selected subset of schema alignment scenarios. We analyzed the changes in the response and the model's performance using different data cardinalities in the prompt (1:1, 1:M, N:1, N:M). In these cases, the left-hand side represents the number of source properties to be mapped, while the right-hand side indicates the number of target classes from which target properties are selected to map the source properties. This approach allowed us to comprehensively examine the model's ability to handle varying complexities in schema alignment tasks, reflecting real-world scenarios where the amount of information in the prompt may vary depending on the entity's preferences. To ensure a more precise evaluation, we ran each prompt multiple times to assess the consistency of the LLM's responses based on the same input. Additionally, we used different depth and threshold values, which also impacted the structure of the final prompt sent to the LLM. The depth defines how many levels of sub-classes, parent-classes, sub-properties, and parent-properties will be included in the mapping process. Meanwhile, the threshold specifies the minimum confidence required for the mappings to be considered valid and returned by the model

| Values       | GPT-4o-Mini 1:1 | GPT-4o-Mini 1:M | GPT-4o-Mini N:1 | GPT-4o-Mini N:M |   | GPT-4o-Mini+ 1:1 | GPT-4o-Mini+ 1:M | GPT-4o-Mini+ N:1 | GPT-4o-Mini+ N:M |
|--------------|-----------------|-----------------|-----------------|-----------------| - |------------------|------------------|------------------|------------------|
| d=1, th=0.5  | **0.601**       | **0.583**       | 0.557           | **0.532**       |   | **0.710**        | **0.722**        | 0.735            | 0.708            |
| d=1, th=0.7  | 0.575           | 0.564           | **0.576**       | 0.476           |   | 0.687            | 0.712            | 0.721            | **0.716**        |
| d=1, th=0.9  | 0.559           | 0.498           | 0.521           | 0.453           |   | 0.668            | 0.697            | **0.746**        | 0.686            |
|              |                 |                 |                 |                 |   |                  |                  |                  |                  |
| d=3, th=0.5  | **0.685**       | 0.562           | **0.512**       | **0.538**       |   | **0.731**        | **0.743**        | 0.723            | **0.740**        |
| d=3, th=0.7  | 0.677           | **0.598**       | 0.497           | 0.502           |   | 0.713            | 0.720            | 0.705            | 0.695            |
| d=3, th=0.9  | 0.618           | 0.534           | 0.458           | 0.429           |   | 0.710            | 0.704            | **0.765**        | 0.711            |
|              |                 |                 |                 |                 |   |                  |                  |                  |                  |
| d=5, th=0.5  | **0.693**       | **0.560**       | **0.530**       | **0.572**       |   | **0.738**        | **0.746**        | 0.692            | 0.722            |
| d=5, th=0.7  | 0.661           | 0.513           | 0.497           | 0.511           |   | 0.719            | 0.713            | **0.714**        | 0.709            |
| d=5, th=0.9  | 0.674           | 0.493           | 0.503           | 0.476           |   | 0.721            | 0.725            | 0.701            | **0.737**        |

In the table, we present the scores from the final analysis, which are calculated as the average of all human evaluation scores for both the classic GPT-4o-Mini model and the fine-tuned GPT-4o-Mini+ model. For the fine-tuned model, we used a subset of generated example mappings, utilizing different contextual BERT embeddings. The findings indicate that increasing the depth of schema exploration generally improves the relevance and order of the generated mappings, highlighting the model's ability to understand deeper contextual information. However, we also observe that, in general, the quality of the results decreases under stricter conditions, such as higher thresholds. In cases where mappings are not feasible, the model tends to hallucinate, generating illogical associations that can distort the results. While the fine-tuned model demonstrated better and more stable performance, these challenges remained. The model’s tendency to hallucinate in these scenarios underscores the need for more advanced training methodologies and sampling strategies.

#### Running locally
The scripts in the `schema_alignment` directory can be used to recreate the experiments conducted in the research, but to also extend and improve the results by integrating additional techniques, models, or combinations of methods to explore new possibilities for schema alignment and verification workflows. The available scripts can be used to:
- preprocess `*_preprocess.py`, merge `merge_ontologies.py` and build `build_ontology.py` ontologies into the data format described in the research.
- generating graph `graph_similarity.py` and context `context_similarity.py` embeddings using different parameters and evaluating the generated embeddings.
- prompting and evaluating schema property mappings using LLM models `prompt_evaluation.py`.
- human evaluating of the generated property mappings based on the scores specified in the research `human_evaluation.py`.

## Contribution

This repository implements the components and services needed to achieve the system described in the research. It provides a starting point for further exploration and development in integrating LLM schema alignment and SSI technologies into data-sharing frameworks. Pull requests are open, and developers are encouraged to contribute. For changes, please fork the repository!

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

This solution is developed as part of my master's thesis at the [University of Ljubljana Faculty of Computer and Information Science](https://www.fri.uni-lj.si/en). I would like to extend my deepest gratitude to:
- My mentor, [Assoc. Prof. dr. Dejan Lavbič](https://www.lavbic.net/), who introduced me to the field of data spaces and provided invaluable guidance and support throughout the research and development of this project.
- The [Veramo](https://github.com/decentralized-identity/veramo) and [iSHARE](https://github.com/iSHAREScheme) communities for their contributions, which significantly optimized the research and development process.
