import numpy as np
import random
import tensorflow as tf
from q_network import QNetwork
from utilities import ExplorationScheduler, init_env, StatsRecorder

def main():
    env = init_env()
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    network = QNetwork(observation_space=env.observation_space.n, action_space=env.action_space.n)
    discount_rate = .99
    num_episodes = 2000
    exploration = ExplorationScheduler(timesteps=num_episodes, start_prob=0.1, end_prob=0.02)
    stats_recorder = StatsRecorder(summary_steps=100, performance_num_episodes=200)

    target_network_update_frequency = 1000
    i = 0
    for episode in range(num_episodes+1):
        observation = env.reset()

        while True:
            action, target_action_values = network.predict_action(observation)

            if random.random() < exploration.value(episode):
                action = env.action_space.sample()

            next_observation, reward, done, info = env.step(action)
            stats_recorder.after_step(reward, done)

            if done:
                target_action_values[action] = reward
            else:
                best_action_values = np.max(network.predict_action_value(next_observation))
                target_action_values[action] = reward + discount_rate * best_action_values


            network.train(observation, [target_action_values])
            observation = next_observation

            if i % target_network_update_frequency == 0:
                network.update()

            i += 1
            if done == True:
                break

if __name__ == "__main__":
    main()