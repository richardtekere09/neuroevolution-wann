import json
import networkx as nx
import matplotlib.pyplot as plt


def visualize_network(filepath):
    with open(filepath, "r") as f:
        structure = json.load(f)

    G = nx.node_link_graph(structure)

    # Group nodes
    input_nodes = sorted([n for n in G.nodes if n.startswith("in_")])
    output_nodes = sorted([n for n in G.nodes if n.startswith("out_")])
    hidden_0_nodes = sorted([n for n in G.nodes if n.startswith("hid_0_")])
    hidden_1_nodes = sorted([n for n in G.nodes if n.startswith("hid_1_")])
    other_hidden_nodes = sorted(
        [
            n
            for n in G.nodes
            if n not in input_nodes + output_nodes + hidden_0_nodes + hidden_1_nodes
            and n.startswith("hid_")
        ]
    )

    # Create positions by x (layer) and y (spread out vertically)
    pos = {}

    for i, node in enumerate(input_nodes):
        pos[node] = (0, -i)

    for i, node in enumerate(hidden_0_nodes):
        pos[node] = (1, -i)

    for i, node in enumerate(hidden_1_nodes):
        pos[node] = (2, -i)

    for i, node in enumerate(other_hidden_nodes):
        pos[node] = (1.5, -i)

    for i, node in enumerate(output_nodes):
        pos[node] = (3, -i)

    # Node colors
    color_map = []
    for node in G.nodes():
        if node.startswith("in_"):
            color_map.append("skyblue")
        elif node.startswith("out_"):
            color_map.append("salmon")
        elif node.startswith("hid_0_"):
            color_map.append("lightgreen")
        elif node.startswith("hid_1_"):
            color_map.append("mediumseagreen")
        else:
            color_map.append("gray")

    # Draw
    plt.figure(figsize=(14, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=800,
        font_size=8,
        arrows=True,
        edge_color="gray",
    )
    plt.title(f"Network Structure: {filepath}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Example call
visualize_network(
    "/Users/richard/neuroevolution-wann/src_improved_WANN/data/generation_0.json"
)
