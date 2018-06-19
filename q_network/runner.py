from q_network.network import Network
from q_network.replay_buffer import ReplayBuffer
from common.utilities import StatsRecorder, ExplorationScheduler

class Runner():

    def __init__(self,
                 env,
                 timesteps,
                 buffer_size,
                 exploration_fraction,
                 exploration_start_prob,
                 exploration_end_prob,
                 target_network_update_frequency,
                 batch_size,
                 observation_space,
                 action_space,
                 learning_rate,
                 discount_rate,
                 training_starts_steps,
                 training_frequency,
                 summary_frequency,
                 performance_num_episodes
                 ):

        self.env = env
        self.timesteps = timesteps

        self.target_network_update_frequency = target_network_update_frequency

        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes)

        self.replay_buffer = ReplayBuffer(buffer_size)
        exploration_scheduler = ExplorationScheduler(timesteps=int(exploration_fraction * timesteps),
                                                         start_prob=exploration_start_prob,
                                                         end_prob=exploration_end_prob)

        self.network = Network(observation_space=observation_space,
                          action_space=action_space,
                          learning_rate=learning_rate,
                          discount_rate=discount_rate,
                          exploration_scheduler=exploration_scheduler
                          )

        self.batch_size = batch_size

        self.training_starts_steps = training_starts_steps
        self.training_frequency = training_frequency

    def run(self):
        self.network.update_target()
        observation = self.env.reset()

        for t in range(self.timesteps):
            action = self.network.predict_action(observation, t)

            next_observation, reward, done, _ = self.env.step(action)
            self.stats_recorder.after_step(reward, done, t)

            self.replay_buffer.add(observation, action, reward, next_observation, done)
            observation = next_observation

            if done:
                observation = self.env.reset()

            if (t > self.training_starts_steps) and (t % self.training_frequency) == 0:
                observations, actions, rewards, next_observations, dones \
                        = self.replay_buffer.sample(self.batch_size)

                self.network.train(observations, actions, rewards, next_observations, dones)
            if (t > self.training_starts_steps) and (t % self.target_network_update_frequency) == 0:
                self.network.update_target()