from evolution import run_evolution, get_top_generations

if __name__ == "__main__":
    # Step 1: Run evolution
    run_evolution(render=False)

    # Step 2: Evaluate all saved generations
    print("\n🔍 Evaluating top generations...")
    top_gens = get_top_generations()

    # Step 3: Display top 3
    print("\n🏆 Top 3 Generations by Fitness:")
    for i, (gen, fitness) in enumerate(top_gens, 1):
        print(f"{i}. Generation {gen} — Fitness: {fitness:.2f}")

    # Step 4: Play/render best agent with shared weights
    best_gen = top_gens[0][0]
    best_path = f"data/generation_{best_gen}.json"
