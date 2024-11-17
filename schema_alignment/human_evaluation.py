import pandas as pd
import json
import os
import re

def evaluate_prompts(type: str) -> None:
    """Execute human evaluation on the generated responses based on different prompts.

    Args:
        type (str): The type of the prompt, i.e. number of source and target components.

    Returns:
         None

    """
    scores = []

    file_path = f'prompts/evaluation_results_{type}.parquet'
    df = pd.read_parquet(file_path)

    unique_configs = df.groupby(['source_attributes', 'target_schemas', 'mapping_type', 'depth', 'threshold'])

    results_csv_path = f'prompts/human_evaluation_results_{type}.csv'
    if os.path.isfile(results_csv_path):
        existing_results = pd.read_csv(results_csv_path)
        evaluated_configs = set(
            existing_results[['source_attributes', 'target_schemas', 'mapping_type', 'depth', 'threshold']].itertuples(
                index=False, name=None))
    else:
        header = ['source_attributes', 'target_schemas', 'mapping_type', 'depth', 'threshold',
                  'mean_relevance_score', 'mean_order_score', 'mean_completeness_score', 'mean_interoperability_score']
        pd.DataFrame(columns=header).to_csv(results_csv_path, index=False)
        evaluated_configs = set()

    response_scores = {}
    last_params = None
    for name, group in unique_configs:
        if name in evaluated_configs:
            print(f"Configuration {name} already evaluated. Skipping...")
            continue

        response_scores = response_scores if last_params == (name[0], name[1]) else {}
        last_params = (name[0], name[1])
        relevance_sum = 0
        order_sum = 0
        completeness_sum = 0
        interoperability_sum = 0

        for index, row in group.iterrows():
            response = row['response'].strip()
            json_pattern = r'json\s*(\{.*?\})\s*'

            # Find the first match (or all matches if multiple JSONs are embedded)
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            # Check if this specific response has already been scored in the current configuration
            if response in response_scores:
                # Retrieve the previous scores for this response
                relevance_score, order_score, completeness_score, interoperability_score = response_scores[response]
                print(f"Using saved scores for response: {response}")
            else:
                print("\n" + "=" * 80)  # Separator line
                print(f"Evaluating response for:")
                print(f"Source Attributes: {row['source_attributes']}")
                print(f"Target Schemas: {row['target_schemas']}")
                print(f"Mapping Type: {row['mapping_type']}")
                print(f"Depth: {row['depth']}")
                print(f"Threshold: {row['threshold']}")
                print("Response: ", response)

                # Get relevance score with validation
                while True:
                    try:
                        relevance_score = int(input("Enter relevance score (1-5): "))
                        if relevance_score in [1, 2, 3, 4, 5]:
                            break
                        else:
                            print("Invalid input! Score must be between 1 and 5. Please try again.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")

                # Get order score with validation
                while True:
                    try:
                        order_score = int(input("Enter order score (1-4): "))  # Changed to 1-3
                        if order_score in [1, 2, 3, 4]:
                            break
                        else:
                            print("Invalid input! Score must be between 1 and 4. Please try again.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")

                # Get completeness score with validation
                while True:
                    try:
                        completeness_score = int(input("Enter completeness score (1-5): "))
                        if completeness_score in [1, 2, 3, 4, 5]:
                            break
                        else:
                            print("Invalid input! Score must be between 1 and 5. Please try again.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")

                # Get interoperability score with validation
                while True:
                    try:
                        interoperability_score = int(input("Enter interoperability score (1-3): "))
                        if interoperability_score in [1, 2, 3]:
                            break
                        else:
                            print("Invalid input! Score must be between 1 and 3. Please try again.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")

                response_scores[response] = (relevance_score, order_score, completeness_score, interoperability_score)

            # Accumulate scores if all inputs are valid
            relevance_sum += relevance_score
            order_sum += order_score
            completeness_sum += completeness_score
            interoperability_sum += interoperability_score

        # Ensure execution count is greater than zero to avoid division by zero
        execution_count = len(group)
        if execution_count > 0:
            mean_relevance = relevance_sum / execution_count
            mean_order = order_sum / execution_count
            mean_completeness = completeness_sum / execution_count
            mean_interoperability = interoperability_sum / execution_count

            # Append the results to the scores list
            scores.append({
                'source_attributes': name[0],
                'target_schemas': name[1],
                'mapping_type': name[2],
                'depth': name[3],
                'threshold': name[4],
                'mean_relevance_score': mean_relevance,
                'mean_order_score': mean_order,
                'mean_completeness_score': mean_completeness,
                'mean_interoperability_score': mean_interoperability,
            })

            # Convert to DataFrame and append to CSV
            scores_df = pd.DataFrame(scores[-1:], columns=scores[-1].keys())
            scores_df.to_csv(results_csv_path, mode='a', header=False, index=False)


def calculate_final_score(type: str) -> None:
    """Calculate the final score based on the human evaluation and different prompt types.

    Args:
        type (str): The type of the prompt, i.e. number of source and target components.

    Returns:
        None

    """
    df = pd.read_csv(f'prompts/human_evaluation_results_{type}.csv')

    max_scores = {
        'mean_relevance_score': 5.0,  # Max score for relevance
        'mean_order_score': 4.0,  # Max score for order
        'mean_completeness_score': 5.0,  # Max score for completeness
        'mean_interoperability_score': 3.0  # Max score for interoperability
    }

    df['final_score'] = 0.0
    for index, row in df.iterrows():
        # Normalize each metric by dividing by the maximum score
        normalized_scores = 0.0
        for metric in ['mean_relevance_score', 'mean_order_score', 'mean_completeness_score',
                       'mean_interoperability_score']:
            normalized_scores += row[metric] / max_scores[metric]

        # Final score is the average of normalized scores
        df.at[index, 'final_score'] = normalized_scores / 4.0

    df.to_csv(f'prompts/human_evaluation_results_{type}_with_final_scores.csv', index=False)


if __name__ == "__main__":
    program_type = "final"
    prompt_type = "N_M" #1_1, 1_M, N_1, N_M

    if program_type == "final":
        calculate_final_score(prompt_type)
    else:
        evaluate_prompts(prompt_type)