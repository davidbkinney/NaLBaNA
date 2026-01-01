from . import initialization
from . import prompts
from . import tools
import ast
import json

#Text-to-graph functions.
def node_extractor(prompt:str) -> list:
    """
    Generates a set of Bayes net nodes based on user input.
    Args:
        prompt: the user prompt describing a causal system.
    Returns:
        List of strings, each of which is a name of a node.
    """
    client = initialization.get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system",
        "content": prompts.NODE_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        tools=[tools.NODE_LIST_TOOL],
        tool_choice="required"
        )
    json_str = response.choices[0].message.tool_calls[0].function.arguments
    j = ast.literal_eval(json_str)
    return j['nodes']

def value_generator(prompt:str,nodes:list) -> list:
    """
    Takes in a list of variables and returns a set of mutually exclusive
    and jointly exhaustive values for those variables.
    Args:
        prompt: the user prompt describing a causal system.
        nodes: a list containing the names of each variable in the 
        causal system extracted by the LLM in an earlier processing stage.
    Returns:
        A list of dictionaries where each dictionary contains the name of a variable 
        and a list of its values.
    """
    client = initialization.get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system",
        "content": prompts.VALUE_GENERATOR_SYSTEM_PROMPT.format(description=prompt)},
            {"role": "user",
            "content": [{"type": "text", "text": str(nodes)}]}
        ],
        tools=[tools.make_value_list_tool(nodes)],
        tool_choice="required"
        )
    tool_calls = response.choices[0].message.tool_calls
    
    # The model may call the tool once; get the arguments JSON
    arguments_json = tool_calls[0].function.arguments
    structured_output = json.loads(arguments_json)

    return structured_output["variables"]

def check_values(value_list:list) -> bool:
    """
    Checks that no variable has less than two possible values.
    Args:
        value_list: A list of dictionaries where each dictionary contains
        the name of a variable and a list of its values.
    Returns:
        False if all variables have more than one value, True otherwise.
    """
    for val in value_list:
        if len(val['values']) < 2:
            return True
    return False
