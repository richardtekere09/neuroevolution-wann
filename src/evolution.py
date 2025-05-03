import gymnasium as gym
import numpy as np
from network import WANN
from utils import evaluate_network, save_network, plot_fitness


def run_evolution(
    generations=20,
    population_size=50,
    weight_candidates=[-2.0, -1.0, 0.0, 1.0, 2.0],
    render=False,
):
    env = gym.make("BipedalWalker-v3", render_mode="human" if render else None)
    population = [WANN() for _ in range(population_size)]

    fitness_history = []

    for gen in range(generations):
        fitness_scores = []

        for i, net in enumerate(population):
            render_this = gen % 5 == 0 and i == 0
            fitness = evaluate_network(net, env, weight_candidates, render=render_this)
            fitness_scores.append(fitness)

        print(
            f"Generation {gen} - Best fitness: {np.max(fitness_scores):.2f}", flush=True
        )
        fitness_history.append(np.mean(fitness_scores))

        # Sort and create new population
        sorted_pairs = sorted(
            zip(fitness_scores, population), key=lambda x: x[0], reverse=True
        )
        elite = [net for _, net in sorted_pairs[: int(0.2 * population_size)]]

        new_population = elite.copy()
        while len(new_population) < population_size:
            parent = np.random.choice(elite)
            child = parent.clone()
            child.mutate()
            new_population.append(child)

        population = new_population
        save_network(population[0], gen)

    plot_fitness(fitness_history)
    env.close()
