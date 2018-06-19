import random
import numpy as np
import q_learning.runner

def main():
    random.seed(0)
    np.random.seed(0)
    runner = q_learning.runner.Runner(timesteps=50000,
                                      exploration_start_prob=1.0,
                                      exploration_end_prob=0.02,
                                      learning_rate=0.1,
                                      discount_rate=0.99,
                                      summary_frequency=5000,
                                      performance_num_episodes=200
                                      )
    runner.run()

if __name__ == '__main__':
    main()
