from common.utilities import ExplorationScheduler, StatsRecorder, init_env
import random
import numpy as np
np.set_printoptions(precision=5)


"""
0 0.0
5000 0.02
10000 0.1
15000 0.15
20000 0.25
25000 0.395
30000 0.52
35000 0.595
40000 0.75
45000 0.9
50000 0.975
"""


class Runner():

    def __init__(self,
                 timesteps,
                 exploration_start_prob,
                 exploration_end_prob,
                 learning_rate,
                 discount_rate,
                 summary_frequency,
                 performance_num_episodes):

        self.env = init_env()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.timesteps = timesteps
        self.exploration_scheduler = ExplorationScheduler(timesteps=timesteps, start_prob=exploration_start_prob, end_prob=exploration_end_prob)
        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes)



    def run(self):
        action_values = np.zeros([self.env.observation_space.n, self.env.action_space.n], dtype=np.float32)
        observation = self.env.reset()

        for i in range(self.timesteps + 1):
            if random.random() < self.exploration_scheduler.value(i):
                action = self.env.action_space.sample()
            else:
                action = np.argmax(action_values[observation, :])

            next_observation, reward, done, info = self.env.step(action)
            self.stats_recorder.after_step(reward, done, i)

            best_action_value = max(action_values[next_observation][:])
            action_value = action_values[observation][action]

            action_values[observation][action] += \
                self.learning_rate * (reward + self.discount_rate * best_action_value - action_value)

            if done:
                observation = self.env.reset()
            else:
                observation = next_observation

        print(np.array2string(action_values, separator=', '))

"""
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
"""