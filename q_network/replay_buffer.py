import random
import numpy as np

class ReplayBuffer():

    def __init__(self, buffer_size):
        self._buffer = []
        self._buffer_size = buffer_size
        self._index = 0

    def add(self, observation, action, reward, next_observation, done):
        transition = (observation, action, reward, next_observation, done)

        if self._index >= len(self._buffer):
            self._buffer.append(transition)
        else:
            self._buffer[self._index] = transition

        if self._index == self._buffer_size -1:
            self._index = 0
        else:
            self._index += 1

    def sample(self, batch_size):
        observations, actions, rewards, next_observations, dones = [], [], [], [], []
        for _ in range(batch_size):
            index = random.randint(0, len(self._buffer) - 1)

            observation, action, reward, next_observation, done = self._buffer[index]
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            dones.append(done)

        return observations, actions, rewards, next_observations, np.array(dones)