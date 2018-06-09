from utilities import ExplorationScheduler, StatsRecorder, init_env
import random
import numpy as np
np.set_printoptions(precision=5)

def main():
    env = init_env()
    random.seed(0)
    discount_rate = .99
    num_episodes = 4000
    exploration = ExplorationScheduler(timesteps=num_episodes, start_prob=0.9, end_prob=0.02)
    stats_recorder = StatsRecorder(summary_steps=100, performance_num_episodes=200)

    action_values = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float32)
    learning_rate = 0.1
    np.random.seed(0)

    for episode in range(num_episodes+1):
        observation = env.reset()
        while True:
            if random.random() < exploration.value(episode):
                action = env.action_space.sample()
            else:
                action = np.argmax(action_values[observation, :])

            next_observation, reward, done, info = env.step(action)
            stats_recorder.after_step(reward, done)

            best_action_value = max(action_values[next_observation][:])
            action_value = action_values[observation][action]

            action_values[observation][action] += \
                learning_rate * (reward + discount_rate * best_action_value - action_value)

            observation = next_observation
            if done == True:
                break

    print(np.array2string(action_values, separator=', '))


if __name__ == "__main__":
    main()