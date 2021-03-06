# Q-learning

## Introduction
Q-learning is a model-free reinforcement learning algorithm. 

A policy is a mapping from states to actions that tells what action
to perform when the environment is in a particular state. An action-value function ```Q```
assigns values to these state-action pairs. Q-learning is a method to learn an 
action-value function by directly approximating the optimal action-value function. A
neural network can be used as a function approximator.

## Environment
We use FrozenLake as an environment. The agent starts at S with the goal to
move to G. The agent can walk over the frozen surface F and needs to avoid
holes H:

![alt text](images/grid_states.png).

The agent can take 4 possible actions:
```
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
```

## Q-learning
The Q-learning implementation is based on [Q-learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).

## Q-network
The Q-network implementation is based on [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf).


A neural network is used as a function approximator to estimate the action-value function

``` python
def action_value_function(inputs, action_space, scope):
    with tf.variable_scope(scope):
        hidden = tf.layers.dense(inputs=inputs, units=64, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        return output
```

The network can be trained by minimising the loss function

``` python
td_errors = selected_action_values - action_values_target
losses = huber_loss(td_errors)
```

## Results
Learned actions using the action-value function after 4000 iterations:

![alt text](images/grid_actions.png).