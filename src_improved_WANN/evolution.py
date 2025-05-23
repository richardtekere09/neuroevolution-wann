import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gymnasium as gym
import numpy as np
from network import WANN
from utils import (
    evaluate_network,
    save_network,
    plot_fitness,
    load_network_from_file,
)

def run_evolution(
    generations=101,
    population_size=50,
    weight_candidates=[-2.0, -1.0, 0.0, 1.0, 2.0],
    render=False,
):
    env = gym.make("BipedalWalker-v3", render_mode="human" if render else None)
    population = [WANN() for _ in range(population_size)]

    fitness_history = []

    for gen in range(generations):
        fitness_scores = [
            evaluate_network(net, env, weight_candidates, render=False)
            for net in population
        ]

        print(
            f"Generation {gen} - Best fitness: {np.max(fitness_scores):.2f}", flush=True
        )
        fitness_history.append(np.mean(fitness_scores))

        # Select top 20% (elite)
        sorted_pairs = sorted(
            zip(fitness_scores, population), key=lambda x: x[0], reverse=True
        )
        elite = [net for _, net in sorted_pairs[: int(0.2 * population_size)]]

        # Generate new population
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent = np.random.choice(elite)
            child = parent.clone()
            child.mutate()
            new_population.append(child)

        population = new_population
        save_network(
            population[0],
            gen,
            folder="/Users/richard/neuroevolution-wann/src_improved_WANN/data",
        )

    plot_fitness(
        fitness_history,
        filename="/Users/richard/neuroevolution-wann/src_improved_WANN/plots/fitness_improved.png",
    )
    env.close()


def get_top_generations(
    folder="/Users/richard/neuroevolution-wann/src_improved_WANN/data",
    shared_weights=[-2.0, -1.0, 0.0, 1.0, 2.0],
    top_n=3,
):
    env = gym.make("BipedalWalker-v3")
    results = []

    for file in os.listdir(folder):
        if file.startswith("generation_") and file.endswith(".json"):
            gen = int(re.findall(r"\d+", file)[0])
            net = load_network_from_file(os.path.join(folder, file))
            fitness = evaluate_network(net, env, shared_weights)
            print(f"Generation {gen} — Fitness: {fitness:.2f}")
            results.append((gen, fitness))

    env.close()
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
