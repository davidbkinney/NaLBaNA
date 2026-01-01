"""
tools.py
Contains tools and functions that create tools used to implement the text-to-bayes-net agent.
"""

NODE_LIST_TOOL = {
            "type": "function",
            "function": {
                "name": "generate_list_of_nodes",
                "description": ("Generates a list of causal variables based on"
                                " user's input. Only outputs string arrays."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": ("A list of causal variables based "
                            "on user's input")
                        }
                    },
                    "required": ["items"]
                }
            }
        }

def make_value_list_tool(variable_names: list) -> dict:
    """
    Creates a tool definition for generating values for given variables.
    Args:
        variable_names: A list of variable names (strings) for which values
                        need to be generated.
    Returns:
        A dictionary defining the tool for use with the OpenAI API.
    """
    if not variable_names:
        raise ValueError("allowed_variables must be a non-empty list of strings")

    if not all(isinstance(v, str) for v in variable_names):
        raise TypeError("All variable names must be strings")
    
    return {
            "type": "function",
            "function": {
                "name": "fill_values_for_variables",
                "description": (
                    "Return a list of dictionaries where each dictionary has "
                    "'variable' (provided in input) and 'values' (a list of possible values). "
                    "You must include every variable from the input exactly once."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variables": {
                            "type": "array",
                            "description": (
                                "A list of dictionaries, each with a 'variable' key. "
                                "The model must fill in the 'values' key for each."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "variable": {"type": "string",
                                                "enum": variable_names},
                                    "values": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "The possible values for this variable."
                                    }
                                },
                                "required": ["variable","values"]
                            }
                        }
                    },
                    "required": ["variables"]
                }
            }
        }

def make_dag_creation_tool(variable_names: list) -> dict:
    """
    Creates a tool definition for generating a directed acyclic graph (DAG)
    over given variables.
    Args:
        variable_names: A list of variable names (strings) that will be the nodes
                        of the DAG.
    Returns:
        A dictionary defining the tool for use with the OpenAI API.
    """
    if not variable_names:
        raise ValueError("variable_names must be a non-empty list of strings")

    if not all(isinstance(v, str) for v in variable_names):
        raise TypeError("All variable names must be strings")
    
    return {
        "type": "function",
        "function": {
            "name": "fill_dag_for_nodes",
            "description": (
                "Given a list of node names, return a directed acyclic graph (DAG) "
                "over those nodes. The model must NOT add new node names, must not "
                "include self-loops, and must produce edges only between the given nodes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {"type": "string",
                                "enum": variable_names},
                        "description": "List of node names (strings)."
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cause": {"type": "string"},
                                "effect": {"type": "string"}
                            },
                            "required": ["cause", "effect"]
                        },
                        "description": "List of directed edges between nodes; each item has 'cause' and 'effect'."
                    }
                },
                "required": ["nodes", "edges"]
            }
        }
    }

def make_conditional_probability_scoring_tool(parent_child_combos:list):
    """
    Creates a tool definition for assigning likelihood scores to conditional
    probability events.
    Args:
        parent_child_combos: A list of dictionaries, each containing an 'event' key
                             (a list of variable-value pairs) and a 'conditions' key
                             (a list of variable-value pairs).
    Returns:
        A dictionary defining the tool for use with the OpenAI API.
    """
    # Collect all unique variable names and values
    variables = set()
    value_map = {}
    for combo in parent_child_combos:
        ev = combo["event"]
        variables.add(ev["variable"])
        value_map.setdefault(ev["variable"], set()).add(ev["value"])
        for cond in combo["conditions"]:
            variables.add(cond["variable"])
            value_map.setdefault(cond["variable"], set()).add(cond["value"])

    # Convert sets to lists
    variables = list(variables)
    for k in value_map:
        value_map[k] = list(value_map[k])

    # Define a reusable variable-value object schema
    def var_val_object():
        return {
            "type": "object",
            "properties": {
                "variable": {"type": "string", "enum": variables},
                "value": {"type": "string"}  
            },
            "required": ["variable", "value"]
        }

    return {
        "type": "function",
        "function": {
            "name": "assign_conditional_likelihood_scores",
            "description": (
                "Assigns a likelihood score from 1 (least likely) to 10 (most likely) "
                "to the event that a variable takes some value, given that other "
                "variables take specific values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conditional_scores_and_probs": {
                        "type": "array",
                        "minItems": len(parent_child_combos),
                        "maxItems": len(parent_child_combos),
                        "items": {
                            "type": "object",
                            "properties": {
                                "event": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 1,
                                    "items": var_val_object()
                                },
                                "conditions": {
                                    "type": "array",
                                    "items": var_val_object()
                                },
                                "score": {
                                    "type": "number",
                                    "minimum": 1,
                                    "maximum": 10
                                }
                            },
                            "required": ["event", "conditions", "score"]
                        }
                    }
                },
                "required": ["conditional_scores_and_probs"]
            }
        }
    }