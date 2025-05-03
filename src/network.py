import random
import copy
import networkx as nx
import numpy as np


class WANN:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.init_structure()

    def init_structure(self):
        # Create input and output nodes
        for i in range(24):  # BipedalWalker input size
            self.graph.add_node(f"in_{i}", type="input")
        for o in range(4):  # BipedalWalker output size
            self.graph.add_node(f"out_{o}", type="output")
        # Add random connections
        for i in range(24):
            for o in range(4):
                if random.random() < 0.2:
                    self.graph.add_edge(f"in_{i}", f"out_{o}")

    def forward(self, inputs, shared_weight):
        outputs = [0.0 for _ in range(4)]
        for i in range(24):
            for o in range(4):
                if self.graph.has_edge(f"in_{i}", f"out_{o}"):
                    outputs[o] += inputs[i] * shared_weight
        return np.tanh(outputs)

    def mutate(self):
        # Randomly add or remove connections
        in_nodes = [n for n in self.graph.nodes if "in_" in n]
        out_nodes = [n for n in self.graph.nodes if "out_" in n]
        i, o = random.choice(in_nodes), random.choice(out_nodes)
        if self.graph.has_edge(i, o):
            self.graph.remove_edge(i, o)
        else:
            self.graph.add_edge(i, o)

    def clone(self):
        clone = WANN()
        clone.graph = copy.deepcopy(self.graph)
        return clone
