import gym
import numpy as np
import random
import tensorflow as tf
from model import Model

"""
1500 0.269
2000 0.361
2500 0.359
3000 0.4
3500 0.41
4000 0.407
4500 0.442
"""
class ExplorationScheduler:
    def __init__(self, timesteps, start_prob, end_prob):
        self.timesteps = timesteps
        self.start_prob = start_prob
        self.end_prob = end_prob

    def value(self, t):
        fraction = min(float(t) / self.timesteps, 1.0)
        return self.start_prob + fraction * (self.end_prob - self.start_prob)

def main():
    env = gym.make('FrozenLake-v0')
    env.seed(0)
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    model = Model(observation_space=16, action_space=4)
    discount_rate = .99
    num_episodes = 1000
    total_rewards = []
    exploration = ExplorationScheduler(timesteps=num_episodes, start_prob=0.1, end_prob=0.02)

    for episode in range(num_episodes):
        observation = env.reset()
        total_reward = 0
        while True:
            action, target_q_value = model.predict_action(observation)

            if random.random() < exploration.value(episode):
                action = env.action_space.sample()

            next_observation, reward, done, info = env.step(action)

            if done:
                target_q_value[0, action] = reward
            else:
                next_q_value = model.predict_q(next_observation)
                target_q_value[0, action] = reward + discount_rate * np.max(next_q_value)

            model.train(observation, target_q_value)
            total_reward += reward
            observation = next_observation

            if done == True:
                break

        if episode % 500 == 0 and episode > 1000:
            score = sum(total_rewards[-1000:]) / 1000.0
            print("{} {}".format(episode, score))

        total_rewards.append(total_reward)

    for i in range(16):
        foo = model.predict_q(i)
        print(foo)

if __name__ == "__main__":
    main()