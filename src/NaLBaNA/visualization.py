import matplotlib.pyplot as plt
import networkx as nx
import pydot 
from typing import List, Dict
import textwrap


def get_layout(G):
    """Helper function to improve graph layout."""
    return nx.spring_layout(G)

def normalize_positions(pos: Dict, margin: float = 0.1) -> Dict:
    """Scale node positions to [margin, 1 - margin] range."""
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(val, min_val, max_val):
        return margin + (1 - 2 * margin) * (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5

    return {
        k: (scale(x, min_x, max_x), scale(y, min_y, max_y))
        for k, (x, y) in pos.items()
    }

def wrap_label(label, width=10):
    "Function to wrap label text."
    return '\n'.join(textwrap.wrap(label, width))

def get_node_size_by_label(G, base_size=1500, scale=100):
    "Function to set the node size for labels."
    return [base_size + scale * len(str(node)) for node in G.nodes]

def plot_causal_dag(relations: List[Dict[str, str]]) -> None:
    """Generate the visualization of a DAG."""
    G = nx.DiGraph()
    for relation in relations:
        G.add_edge(relation['cause'], relation['effect'])

    raw_pos = get_layout(G)
    pos = normalize_positions(raw_pos, margin=0.1)

    fig, ax = plt.subplots(figsize=(10, 8))

    node_sizes = get_node_size_by_label(G)

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels={n: wrap_label(n, 12) for n in G.nodes},
        arrows=True,
        node_size=node_sizes,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        edge_color='gray',
        arrowstyle='-|>',
        arrowsize=15,
    )

    ax.set_title("Causal DAG")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

    plt.show()