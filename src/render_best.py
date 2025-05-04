import gymnasium as gym
import json
import networkx as nx
import numpy as np
from time import sleep
from network import WANN


def load_network(path):
    net = WANN()
    with open(path, "r") as f:
        structure = json.load(f)
    net.graph = nx.node_link_graph(structure)
    return net


def render_with_weights(net, shared_weights, delay=2):
    env = gym.make("BipedalWalker-v3", render_mode="human")

    for w in shared_weights:
        print(f"\nRendering with shared weight: {w}")
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = net.forward(obs, w)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Shared weight {w} -> Total reward: {total_reward:.2f}")
        sleep(delay)  # short pause between runs

    env.close()


if __name__ == "__main__":
    net = "data/generation_76.json"
    # Load your best model (e.g., generation_92)
    best_net = load_network(net)
    print(f"\n🎬 Playing best agent from generation {net}...\n")
    # Define the shared weights to test
    shared_weights = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # Render all
    render_with_weights(best_net, shared_weights)
