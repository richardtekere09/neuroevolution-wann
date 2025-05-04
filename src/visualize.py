import json
import networkx as nx
import matplotlib.pyplot as plt


def visualize_network(filepath):
    # Load the saved graph structure
    with open(filepath, "r") as f:
        structure = json.load(f)

    G = nx.node_link_graph(structure)

    # Get node colors based on type
    color_map = []
    for node in G.nodes(data=True):
        node_type = node[1].get("type", "")
        if "in_" in node[0]:
            color_map.append("skyblue")
        elif "out_" in node[0]:
            color_map.append("salmon")
        else:
            color_map.append("lightgreen")

    # Draw the graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)  # for consistent layout
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
    plt.tight_layout()
    plt.show()


visualize_network("data/generation_0.json")
