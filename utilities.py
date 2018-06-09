from gym.envs.registration import register
import gym

class ExplorationScheduler:
    def __init__(self, timesteps, start_prob, end_prob):
        self.timesteps = timesteps
        self.start_prob = start_prob
        self.end_prob = end_prob

    def value(self, t):
        fraction = min(float(t) / self.timesteps, 1.0)
        return self.start_prob + fraction * (self.end_prob - self.start_prob)

def init_env():
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )

    env = gym.make('FrozenLakeNotSlippery-v0')
    env.seed(0)
    return env

class StatsRecorder():

    def __init__(self, summary_steps, performance_num_episodes):
        self.total_rewards = []
        self.summary_steps = summary_steps
        self.performance_num_episodes = performance_num_episodes
        self.total_reward = 0
        self.num_episodes = 0

    def after_step(self, reward, done):
        self.total_reward += reward
        if done:
            if self.num_episodes % self.summary_steps == 0 and self.num_episodes > self.performance_num_episodes:
                score = sum(self.total_rewards[-self.performance_num_episodes:]) / self.performance_num_episodes
                print("{} {}".format(self.num_episodes, score))

            self.num_episodes += 1

            self.total_rewards.append(self.total_reward)
            self.total_reward = 0