import json
import networkx as nx
import matplotlib.pyplot as plt


def visualize_network(filepath):
    with open(filepath, "r") as f:
        structure = json.load(f)

    G = nx.node_link_graph(structure)

    # Group nodes by type
    input_nodes = sorted([n for n in G.nodes if n.startswith("in_")])
    output_nodes = sorted([n for n in G.nodes if n.startswith("out_")])
    hidden_nodes = sorted([n for n in G.nodes if n not in input_nodes + output_nodes])

    # Create positions
    pos = {}

    # Space nodes vertically (y) and fix x-coordinates
    for i, node in enumerate(input_nodes):
        pos[node] = (0, -i)

    for i, node in enumerate(hidden_nodes):
        pos[node] = (1, -i)

    for i, node in enumerate(output_nodes):
        pos[node] = (2, -i)

    # Node color
    color_map = []
    for node in G.nodes():
        if node.startswith("in_"):
            color_map.append("skyblue")
        elif node.startswith("out_"):
            color_map.append("salmon")
        else:
            color_map.append("lightgreen")

    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=800,
        font_size=8,
        arrows=True,
    )
    plt.title(f"Network Structure: {filepath}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Example call
visualize_network("data/generation_100.json")
