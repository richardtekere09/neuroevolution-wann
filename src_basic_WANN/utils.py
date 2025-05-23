import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from network import WANN

def evaluate_network(net, env, weight_values, render=False):
    best_reward = float("-inf")
    for w in weight_values:
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = net.forward(obs, w)
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            total_reward += reward
            done = terminated or truncated
        best_reward = max(best_reward, total_reward)
    return best_reward


def save_network(net, generation, folder="data"):
    os.makedirs(folder, exist_ok=True)
    structure = nx.node_link_data(net.graph)
    with open(os.path.join(folder, f"generation_{generation}.json"), "w") as f:
        json.dump(structure, f)


def plot_fitness(fitness_history, filename="plots/fitness.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Basic WANN Evolution Convergence\n(Базовая конвергенция WANN)")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def load_network_from_file(filepath):
    """Load a WANN network from a saved JSON file."""
    with open(filepath, "r") as f:
        structure = json.load(f)
    net = WANN()
    net.graph = nx.node_link_graph(structure)
    return net