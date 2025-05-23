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


def save_network(network, generation, folder="src_improved_WANN/data"):
    # Always save relative to the current file (e.g. src_improved_WANN/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(
        base_dir, "..", folder
    )  # go up from shared/, into correct folder
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"generation_{generation}.json")
    with open(filename, "w") as f:
        json.dump(nx.node_link_data(network.graph), f)


def plot_fitness(
    fitness_history,
    filename="/Users/richard/neuroevolution-wann/src_improved_WANN/plots",
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title(
        "Improve WANN Evolution Convergence\n(Улучшенный алгоритм конвергенция WANN)"
    )
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

