"""
graphing.py

Contains functions the create DAGs relating the variables extracted 
from the user prompt, based on the user description of a causal 
system.
"""
from. import initialization, prompts, tools
import json
import ast
import collections 
from collections import defaultdict


def find_cycles(edges:list):
    """Detects any cycles in a list of edges."""
    # Build adjacency list
    graph = defaultdict(list)
    for edge in edges:
        graph[edge['cause']].append(edge['effect'])

    all_cycles = []
    visited = set()

    def dfs(node, stack):
        if node in stack:
            cycle_start = stack.index(node)
            cycle_nodes = stack[cycle_start:] + [node]
            cycle_edges = []
            for i in range(len(cycle_nodes) - 1):
                cycle_edges.append({'cause': cycle_nodes[i],
                                    'effect': cycle_nodes[i+1]})
            all_cycles.append(cycle_edges)
            return
        if node in visited:
            return

        visited.add(node)
        stack.append(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor, stack)
        stack.pop()

    # Important: prevent dict size change during iteration
    nodes = list(graph.keys())
    for edge in edges:
        if edge['effect'] not in graph:
            nodes.append(edge['effect'])

    # Run DFS from each node
    for node in set(nodes):
        dfs(node, [])

    return all_cycles


def nodes_from_graph(graph:list) -> list:
  """Extracts a list of nodes from a graph."""
  nodes = set()
  for edge in graph:
    nodes.add(edge['cause'])
    nodes.add(edge['effect'])
  return list(nodes)


def cycle_breaker(cycle:list) -> list:
  """
  Uses the LLM to undo any cycles.
  Args:
      cycle: list of edges producing a cycle.
  Returns:
      list of edges with the cycle broken.
  """
  nodes = nodes_from_graph(cycle)
  client = initialization.get_client()
  response = client.chat.completions.create(
      model="gpt-4.1",
      messages=[{"role": "system",
      "content": prompts.CYCLE_BREAKER_SYSTEM_PROMPT},
          {"role": "user", "content": str(cycle)}
      ],
      tools=[tools.make_dag_creation_tool(nodes)],
      tool_choice="required"
    )
  json_str = response.choices[0].message.tool_calls[0].function.arguments
  json = ast.literal_eval(json_str)
  return json['edges']


def graph_generator(prompt:str,nodes:list) -> dict:
    """
    Given a user prompt and a list of nodes, returns a graph defined over those nodes
    in keeping with the user input.
    Args:
        prompt: the user prompt describing a causal system.
        nodes: a list of nodes in the causal system.
    Returns:
        a list of edges defined over nodes.
    """
    client = initialization.get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system",
        "content": prompts.GRAPH_GENERATOR_SYSTEM_PROMPT.format(description=prompt)},
            {"role": "user",
            "content": [{"type": "text", "text": str(nodes)}]} # Modified
        ],
        tools=[tools.make_dag_creation_tool(nodes)],
        tool_choice="required"
        )
    tool_calls = response.choices[0].message.tool_calls
    
    # The model may call the tool once; get the arguments JSON
    arguments_json = tool_calls[0].function.arguments
    structured_output = json.loads(arguments_json)
    if len(find_cycles(structured_output['edges']))==0:
        return structured_output['edges']
    else:
        return cycle_breaker(structured_output['edges'])
    
    
def get_parents(node:str,graph:list) -> list:
    """
    Given a node and a graph, returns the parents of that node.
    Args:
        node: the node to find parents for.
        graph: the graph to search.
    Returns:
        a list of parent nodes.
    """
    parents = []
    for edge in graph:
        if edge['effect'] == node:
            parents.append(edge['cause'])
    return parents


def nodes_checker(variables:list, edges:list) -> bool:
    """
    Checks that all nodes in the edges list are present in the variables list.
    Args:
        variables: A list of variable names.
        edges: A list of dictionaries where each dictionary contains
        a 'from' and 'to' key representing an edge in the graph.
    Returns:
        False if all nodes in edges are present in variables, True otherwise.
    """
    causes = [e['cause'] for e in edges]
    effects = [e['effect'] for e in edges]
    all_nodes = set(causes + effects)
    if all_nodes == set(variables):
        return False
    else:
        return True

