import random
import numpy as np
import tensorflow as tf
from common.utilities import init_env
import time
from q_network.runner import Runner

def main():
    random.seed(0)
    tf.set_random_seed(0)
    np.random.seed(0)
    env = init_env()
    start = time.time()

    runner = Runner(env,
                    timesteps=8000,
                    buffer_size=50000,
                    exploration_fraction=0.8,
                    exploration_start_prob=1.0,
                    exploration_end_prob=0.0,
                    target_network_update_frequency=500,
                    batch_size=32,
                    observation_space=16,
                    action_space=4,
                    learning_rate=0.001,
                    discount_rate=0.99,
                    training_starts_steps=1000,
                    training_frequency=1,
                    summary_frequency=500,
                    performance_num_episodes=100
                    )
    runner.run()
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    main()
