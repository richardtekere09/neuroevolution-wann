import os
import re
from evolution import run_evolution, play_best_agent

if __name__ == "__main__":
    run_evolution(render=False)

    # Automatically find the highest generation number
    generation_files = [
        f
        for f in os.listdir("data")
        if f.startswith("generation_") and f.endswith(".json")
    ]
    generation_nums = [int(re.findall(r"\d+", f)[0]) for f in generation_files]
    best_generation = max(generation_nums)
    best_filepath = f"data/generation_{best_generation}.json"

    print(f"Playing best agent from generation {best_generation}...\n")
    play_best_agent(best_filepath, [-2.0, -1.0, 0.0, 1.0, 2.0])
