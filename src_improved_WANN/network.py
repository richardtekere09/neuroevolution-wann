import random
import copy
import networkx as nx
import numpy as np

MAX_HIDDEN_NODES = 8


class WANN:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.hidden_count = MAX_HIDDEN_NODES
        self.init_structure()

    def init_structure(self):
        # Add input and output nodes
        for i in range(24):
            self.graph.add_node(f"in_{i}", type="input")
        for o in range(4):
            self.graph.add_node(f"out_{o}", type="output")
        # Add fixed hidden nodes
        for h in range(MAX_HIDDEN_NODES):
            self.graph.add_node(f"hid_{h}", type="hidden")
        # Add sparse initial connections with preference for hidden→output
        all_nodes = [f"in_{i}" for i in range(24)] + [
            f"hid_{h}" for h in range(MAX_HIDDEN_NODES)
        ]
        target_nodes = [f"hid_{h}" for h in range(MAX_HIDDEN_NODES)] + [
            f"out_{o}" for o in range(4)
        ]
        for _ in range(60):
            if random.random() < 0.4:
                a = random.choice([f"hid_{h}" for h in range(MAX_HIDDEN_NODES)])
                b = random.choice([f"out_{o}" for o in range(4)])
            else:
                a = random.choice(all_nodes)
                b = random.choice(target_nodes)
            if a != b and not self.graph.has_edge(a, b):
                if not nx.has_path(self.graph, b, a):
                    self.graph.add_edge(a, b)

    def forward(self, inputs, shared_weight):
        values = {f"in_{i}": inputs[i] for i in range(24)}
        try:
            nodes_sorted = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            print("⚠️ Skipping network due to cycle.")
            return [0.0] * 4  # fail-safe output

        for n in nodes_sorted:
            if n.startswith("in_"):
                continue
            inputs_sum = sum(
                values.get(src, 0.0) * shared_weight
                for src in self.graph.predecessors(n)
            )
            values[n] = np.tanh(inputs_sum)

        return [np.clip(values.get(f"out_{o}", 0.0), -1, 1) for o in range(4)]

    def mutate(self):
        mutation_type = random.choice(["add_conn", "remove_conn"])
        if mutation_type == "add_conn":
            self._add_connection()
        elif mutation_type == "remove_conn":
            self._remove_connection()

    def _add_connection(self):
        nodes = list(self.graph.nodes)
        attempts = 10
        for _ in range(attempts):
            if random.random() < 0.3:
                a = random.choice(
                    [n for n in nodes if self.graph.nodes[n]["type"] == "hidden"]
                )
                b = random.choice(
                    [n for n in nodes if self.graph.nodes[n]["type"] == "output"]
                )
            else:
                a, b = random.sample(nodes, 2)
            if a == b or self.graph.has_edge(a, b):
                continue
            if self.graph.nodes[a]["type"] == "output":
                continue
            if self.graph.nodes[b]["type"] == "input":
                continue
            if nx.has_path(self.graph, b, a):
                continue
            self.graph.add_edge(a, b)
            return

    def _remove_connection(self):
        edges = list(self.graph.edges)
        if edges:
            self.graph.remove_edge(*random.choice(edges))

    def clone(self):
        clone = WANN()
        clone.graph = copy.deepcopy(self.graph)
        clone.hidden_count = self.hidden_count
        return clone
