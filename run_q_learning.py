from utilities import ExplorationScheduler, StatsRecorder, init_env
import random
import numpy as np
np.set_printoptions(precision=5)

def main():
    env = init_env()
    random.seed(0)
    np.random.seed(0)

    discount_rate = .99
    learning_rate = 0.1
    timesteps = 50000

    exploration = ExplorationScheduler(timesteps=timesteps, start_prob=1.0, end_prob=0.02)
    stats_recorder = StatsRecorder(summary_frequency=5000, performance_num_episodes=200)
    action_values = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float32)
    observation = env.reset()

    for i in range(timesteps+1):
        if random.random() < exploration.value(i):
            action = env.action_space.sample()
        else:
            action = np.argmax(action_values[observation, :])

        next_observation, reward, done, info = env.step(action)
        stats_recorder.after_step(reward, done, i)

        best_action_value = max(action_values[next_observation][:])
        action_value = action_values[observation][action]

        action_values[observation][action] += \
            learning_rate * (reward + discount_rate * best_action_value - action_value)

        if done:
            observation = env.reset()
        else:
            observation = next_observation

    print(np.array2string(action_values, separator=', '))

if __name__ == "__main__":
    main()