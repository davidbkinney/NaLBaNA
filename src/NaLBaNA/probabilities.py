"""
probabilities.py

Contains functions that allow the generation of conditional probability tables
defining probabilities for each variable in the LLM-generated DAG, given each
set of possible values for its parents.
"""

from . import initialization, prompts, tools
from itertools import product
import ast 
import numpy as np


def softmax(score:int, scores:list, beta:float) -> float:
    """Convert a score to a probability using softmax."""
    exp_scores = np.exp(np.array(scores) * beta)
    probabilities = exp_scores / np.sum(exp_scores)
    index = scores.index(score)
    return float(probabilities[index])


def get_joint_combos(values, fixed_probs=None):
    """
    Get all combinations of value assignments for variables.
    """
    var_names = [v['variable'] for v in values]
    value_lists = [v['values'] for v in values]
    
    all_combos = []
    for combo in product(*value_lists):
        all_combos.append(
            [{'variable': var, 'value': val} for var, val in zip(var_names, combo)]
        )

    return all_combos


def get_parent_child_combos(child:str, parents:list, values:list) -> list:
    """Generates all parent-child combinations for a given child node."""
    child_values = [v for v in values if v['variable'] == child][0]
    parent_values = [v for v in values if v['variable'] in parents]

    all_parent_value_combos = []

    parent_value_lists = [
        [pv for pv in parent_values if pv['variable'] == p][0]['values']
        for p in parents
    ]

    for v in child_values['values']:
        for parent_combo in product(*parent_value_lists):
            dictionary = {
                "event": {"variable": child, "value": v},
                "conditions": []
            }

            for p, val in zip(parents, parent_combo):
                dictionary["conditions"].append({"variable": p, "value": val})

            all_parent_value_combos.append(dictionary)

    return all_parent_value_combos


def probability_scores(variable: str, prompt:str, combos:list, max_retries=3) -> list:
    """
    Assigns a conditional probability score between 1 and 10 to each even that a variables takes
    any of its values, given that some other variables take specific values.

    Args:
        variable: the variable for which to assign scores.
        prompt: the user prompt describing a causal system.
        combos: a list of all variable-value combinations for all variables being conditioned upon.
        max_retries: maximum number of retries for getting a valid response.

    Returns:
        A dictionary containing the variable name and a list consisting of all variable-value combinations
        being assigned a conditional probability score, the variable-value combinations being conditioned upon,
        and the assigned score.
    """
    client = initialization.get_client()
    attempts = 0

    while attempts < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": prompts.CONDITIONAL_PROBABILITY_SCORING_SYSTEM_PROMPT.format(description=prompt)},
                    {"role": "user", "content": str(combos)}
                ],
                tools=[tools.make_conditional_probability_scoring_tool(combos)],
                tool_choice="required"
            )
        
            # Extract and parse the JSON string from the tool call
            json_str = response.choices[0].message.tool_calls[0].function.arguments
            j = ast.literal_eval(json_str)
            j['variable'] = variable

            # Get the list of combinations, defaulting to None if key is missing
            scored_combos = j.get("conditional_scores_and_probs")

            # --- Validation Logic ---
            # Check 1: Ensure 'scored_combos' is a non-empty list.
            if not isinstance(scored_combos, list) or not scored_combos:
                print(f"Attempt {attempts + 1} failed: 'conditional_scores_and_probs' is missing, not a list, or empty. Retrying...")
                attempts += 1
                continue

            # Check 2: Retry if EVERY item in the list is missing the 'score' key.
            if all('score' not in item for item in scored_combos):
                print(f"Attempt {attempts + 1} failed: No items contained a 'score' key. Retrying...")
                attempts += 1
                continue
        
            # If all checks pass, the response is valid.
            return j

        except (IndexError, KeyError, SyntaxError, ValueError) as e:
            # Catch potential errors from a malformed API response or parsing failure.
            print(f"Attempt {attempts + 1} failed with an error: {e}. Retrying...")
            attempts += 1
  
    # If all retries fail, raise an exception to signal the failure.
    raise Exception(f"Failed to get a valid response after {max_retries} attempts. JSON String: {json_str}")


def assign_conditional_probabilities(scores:dict,beta:float) -> dict:
    """
    Converts conditional probability scores to conditional probabilities using softmax.
    Args:
        scores: A dictionary containing the variable name and a list consisting of all variable-value combinations
                being assigned a conditional probability score, the variable-value combinations being conditioned upon,
                and the assigned score.
        beta: The beta parameter for the softmax function.
    
    Returns:
        The input dictionary with an added 'conditional_probability' key for each score entry.
    """
    #Find the unique conditions in the input dictionary.
    unique_conditions = list(set(tuple((c['variable'], c['value']) for c in s['conditions'])
        for s in scores['conditional_scores_and_probs']))
    
    #For each unique condition, compute softmax probabilities for the associated probability 
    #scores assigned to each event, given that condition, and add them to the output dictionary. 
    for c in unique_conditions:
        conditional_scores = [s['score'] for s in scores['conditional_scores_and_probs'] if 
                              tuple((c['variable'], c['value']) for c in s['conditions']) == c]
        for s in scores['conditional_scores_and_probs']:
            if tuple((c['variable'], c['value']) for c in s['conditions']) == c:
                s['conditional_probability'] = softmax(s['score'], conditional_scores, beta)
    # Return the updated dictionary with conditional probabilities.
    return scores 
