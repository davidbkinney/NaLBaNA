"""
bayes_net.py

Contains the core code to generate Bayesian networks from prompts.
"""
from . import probabilities, variables, graphing, visualization, initialization

from dataclasses import dataclass
import numpy as np
import pandas as pd

def input_key(key:str):
    """Set up the OpenAI client."""
    initialization.openai_setup(key)

#Create the BayesNet dataclass.
@dataclass
class BayesNet:
    vars:list
    values:dict
    graph:list
    conditional_probabilities:list

def generate_bayes_net(prompt:str) -> BayesNet:
    """
    Generates a Bayesian network from a prompt.

    Args:
        prompt: A user prompt describing a causal system.

    Returns:
        A BayesNet dataclass instance representing the generated Bayesian network.
    """
    print("Extracting variables.")
    #Prompt GPT 4 to create the variable list.
    var_list = variables.node_extractor(prompt)

    #Check the variable list contains at least two values.
    while len(var_list) < 2:
        print("Error: Less than two variables extracted. Retrying variable extraction.")
        var_list = variables.node_extractor(prompt)
    print("Variables extracted.")
    print("Generating variable values.")

    #Prompt GPT4 to generate a list of values for each variable.
    values = variables.value_generator(prompt, var_list)

    #Check that the all and only the previously generated variables are assigned values.
    while [v["variable"] for v in values] != var_list:
        print("Error: Mismatch between extracted variables and generated variable values. Retrying value generation.")
        values = variables.value_generator(prompt, var_list)

    #Check that all variables have at least two values.
    while variables.check_values(values):
        print("Error: One or more variables have invalid values. Retrying value generation.")
        values = variables.value_generator(prompt, var_list)
    print("Variable values generated.")

    #Prompt GPT4 to define a DAG over the causal variables.
    print("Defining causal graph.")
    graph_list = graphing.graph_generator(prompt, [v['variable'] for v in values])

    #Check that all and only the previously generated variables are included in the DAG.
    while graphing.nodes_checker([v['variable'] for v in values], graph_list):
        print("Error: Graph contains nodes not in variable list. Retrying graph generation.")
        graph_list = graphing.graph_generator(prompt, [v['variable'] for v in values])

    #For each variable in the DAG, prompt GPT4 to assign a probability score to each possible
    #value of that variable, given each combination of values taken by its parents. Then convert
    #those scores into probabilities via softmax.
    cond_probs = []
    count = 1
    for var in var_list:
        parents = graphing.get_parents(var, graph_list)
        combos = probabilities.get_parent_child_combos(var, parents, values)
        scores = probabilities.probability_scores(var, prompt, combos)
        conditional_probs = probabilities.assign_conditional_probabilities(scores, beta=.2)
        cond_probs.append(conditional_probs)
        print(f"{count} of {len(var_list)} conditional probability tables generated (one per variable).")
        count += 1
    print("Conditional probability tables generated.")

    #Create the Bayesian Network object.
    bayes_net = BayesNet(var_list,values,graph_list,cond_probs)
    print("Bayesian network generated!")
    return (bayes_net)


def visualize(bayes_net:BayesNet) -> None:
    """Generates a visualization of the Bayesian network."""
    visualization.plot_causal_dag(bayes_net.graph)


def get_joint_distribution(bayes_net:BayesNet,intervention=None) -> pd.DataFrame:
    """
    Computes the joint distribution of the Bayesian network.
    
    Args:
        bayes_net: The BayesNet dataclass instance.
        intervention: A list of interventions to apply (optional).

    Returns:
        A pandas DataFrame representing the joint distribution, applying
        any specified interventions.
    """
    #Obtain the Cartesian product of the value space of each variable.
    joint_combos = probabilities.get_joint_combos(bayes_net.values)
    
    #Create one row of the dataframe for each element of the Cartesian product.
    rows = [{c['variable']: c['value'] for c in combo} for combo in joint_combos]

    #Loop through each element of the Cartesian product, and calculate its joint probability
    #by looping through each variable and obtaining the conditional probability of it taking
    #the value in the product, given that its parents in the obtained DAG take their value in the
    #product.
    for row in rows:
        probs = []
        for var in bayes_net.vars:
            cpt = [cp for cp in bayes_net.conditional_probabilities if cp['variable'] == var][0]
            variable_value_matches = [e for e in cpt['conditional_scores_and_probs'] if e['event'][0]['value'] == row[var]]
            parents = graphing.get_parents(var, bayes_net.graph)
            parent_values = [{'variable': p, 'value': row[p]} for p in parents]
            parent_value_match = [
                e for e in variable_value_matches
                if (
                    (not parents and not e['conditions']) or
                    sorted(e['conditions'], key=lambda x: x['variable'])
                    == sorted(parent_values, key=lambda x: x['variable'])
                )
            ][0]
            #If an intervention has been specified, enforce the logic of the do-calculus
            #when calculating joint the joint distribution.
            if intervention is not None:
                matching = [i for i in intervention if i['variable'] == var]
                if matching:
                    probs.append(1.0 if matching[0]['value'] == row[var] else 0.0)
                else:
                    probs.append(parent_value_match['conditional_probability'])
            else:
                    probs.append(parent_value_match['conditional_probability'])
            
        #Obtain the joint probability by finding the product of all the conditional
        #probabilities for children, given their parents.
        row['joint_probability'] = np.prod(probs)
    joint_df = pd.DataFrame(rows)
    return joint_df

def get_conditional_probability_table(bayes_net:BayesNet, event_variable:str, condition_variables:list,
                                      intervention=None) -> pd.DataFrame:
    """
    Obtains the conditional probability table P(event_variable | condition_variables) from the Bayes net.

    Args:
        bayes_net: The Bayes net to analyze.
        event_variable: The variable for which to compute the conditional probabilities.
        condition_variables: A list of variables to condition on.
        intervention: A list of interventions to apply (optional).

    Returns:
        A pandas DataFrame representing the conditional probability table.
    """
    joint_df = get_joint_distribution(bayes_net, intervention)
    column_names = [' ']
    condition_values = [v for v in bayes_net.values if v['variable'] in condition_variables]
    condition_combos = probabilities.get_joint_combos(condition_values)
    event_values = [v for v in bayes_net.values if v['variable'] == event_variable][0]['values']
    rows = []
    for val in event_values:
        dictionary = {' ': val}
        for cond_combo in condition_combos:
            dictionary_key = " ".join([f"{c['variable']}={c['value']}" for c in cond_combo])
            numerator = np.sum([r['joint_probability'] for _, r in joint_df.iterrows() if 
                                all(r[c['variable']] == c['value'] for c in cond_combo) and r[event_variable] == val])
            denominator = np.sum([r['joint_probability'] for _, r in joint_df.iterrows() if 
                                all(r[c['variable']] == c['value'] for c in cond_combo)])
            dictionary[dictionary_key] = numerator / denominator if denominator > 0 else 0.0
        rows.append(dictionary)
    cond_prob_df = pd.DataFrame(rows)
    return cond_prob_df


def get_marginal_distribution(bayes_net:BayesNet, variable:str, intervention=None) -> pd.DataFrame:
    """
    Obtains the marginal distribution of a variable in a Bayes net.

    Args:
        bayes_net: The Bayes net to analyze.
        variable: The variable for which to compute the marginal distribution.
        intervention: A list of interventions to apply (optional).  

    Returns:
        A pandas DataFrame representing the marginal distribution of the variable.
    """
    joint_df = get_joint_distribution(bayes_net, intervention)
    marginal_rows = []
    variable_values = [v for v in bayes_net.values if v['variable'] == variable][0]['values']
    for val in variable_values:
        prob = np.sum([r['joint_probability'] for _, r in joint_df.iterrows() if r[variable] == val])
        marginal_rows.append({variable: val, 'marginal_probability': prob})
    marginal_df = pd.DataFrame(marginal_rows)
    return marginal_df

def change_conditional_probabilities(bayes_net:BayesNet, variable:str, 
                                     parent_values:list, new_probabilities:dict) -> BayesNet:
    """
    Changes the conditional probabilities of a variable in a Bayes net, given some combination
    of values for its parents.

    Args:
        bayes_net: The Bayes net to modify.
        variable: The name of the variable whose conditional probabilities to change.
        parent_values: A list of dictionaries, each containing a 'variable' and 'value' key, 
        where the variable is a parent of the target variable and the value is the specific value of 
        that parent.
        new_probabilities: A list of dictionaries, each containing a 'value' and 'new_probability' key.

    Returns:
        The modified Bayes net.
    """
    parents = graphing.get_parents(variable, bayes_net.graph)
    input_variables = [par["variable"] for par in parent_values]
    if set(parents) != set(input_variables):
        raise ValueError("Input parent variables do not match actual parents of the variable.")
    for val in parent_values:
        values = [v for v in bayes_net.values if v['variable'] == val['variable']][0]['values']
        if val['value'] not in values:
            raise ValueError(f"{val['value']} is not a value of the variable {val['variable']}.")
    conditional_probabilities = bayes_net.conditional_probabilities
    for cp in conditional_probabilities:
        if cp['variable'] == variable:
            for entry in cp['conditional_scores_and_probs']:
                if entry['conditions'] == parent_values:
                    entry['score'] = "Human-assigned probability replaces LLM-assigned score."
                    entry['conditional_probability'] = [n["new_probability"] for n in new_probabilities if n['value'] == entry['event'][0]['value']][0]
    bayes_net.conditional_probabilities = conditional_probabilities
    return bayes_net

    
            
    
