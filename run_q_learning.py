from utilities import ExplorationScheduler, StatsRecorder, init_env
import random
import numpy as np

def main():
    env = init_env()
    random.seed(0)
    discount_rate = .99
    num_episodes = 2000
    exploration = ExplorationScheduler(timesteps=num_episodes, start_prob=0.9, end_prob=0.02)
    stats_recorder = StatsRecorder(summary_steps=100, performance_num_episodes=200)

    action_values = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float32)
    learning_rate = 0.1
    np.random.seed(0)

    for episode in range(num_episodes):
        observation = env.reset()
        while True:
            if random.random() < exploration.value(episode):
                action = env.action_space.sample()
            else:
                action = np.argmax(action_values[observation, :])

            next_observation, reward, done, info = env.step(action)
            stats_recorder.after_step(reward, done)

            action_values[observation][action] += learning_rate * \
                                                  (reward +
                                                   discount_rate * max(action_values[next_observation][:])
                                                   - action_values[observation][action])

            observation = next_observation
            if done == True:
                break


if __name__ == "__main__":
    main()