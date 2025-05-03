import gymnasium as gym


def run_bipedal_walker():
    # Create the environment
    env = gym.make(
        "BipedalWalker-v3", render_mode="human"
    )  # Use "rgb_array" if you don't want live rendering

    for episode in range(3):  # Run 3 test episodes
        observation, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()  # Optional: show the environment
            action = (
                env.action_space.sample()
            )  # Random action (you'll replace this with WANN later)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1} total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    run_bipedal_walker()
