import tensorflow as tf
import numpy as np
import random

def huber_loss(x):
    return tf.where(
        tf.abs(x) < 1,
        tf.square(x) * 0.5,
        (tf.abs(x) - 0.5)
    )

def action_value_function(inputs, action_space, scope):
    with tf.variable_scope(scope):
        hidden = tf.layers.dense(inputs=inputs, units=64, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        return output

def variables_with_scope(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

class Network():

    def __init__(self, observation_space,
                 action_space,
                 learning_rate,
                 discount_rate,
                 exploration_scheduler,
                 clip_norm=10
                 ):
        self.session = tf.Session()
        self.observation_space = observation_space
        self.action_space = action_space
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.discount_rate = discount_rate
        self.clip_norm = clip_norm
        self.build()
        self.session.run(tf.global_variables_initializer())
        self.exploration_scheduler = exploration_scheduler


    def predict_action(self, observations, t):
        if random.random() < self.exploration_scheduler.value(t):
            return random.randint(0, self.action_space-1)
        else:
            best_action = self.session.run(self.best_action, feed_dict={self.observations: [observations]})
            return best_action

    def update_target(self):
        assign_operations = []
        action_value_variables = sorted(self.action_value_variables, key=lambda variable: variable.name)
        action_value_target_variables = sorted(self.action_value_target_variables, key=lambda variable: variable.name)

        for variables, variables_target in zip(action_value_variables, action_value_target_variables):
            assign_operations.append(variables_target.assign(variables))
        self.session.run(assign_operations)

    def train(self, observations, actions, rewards, next_observations, dones):
        action_values_target_network = self.session.run(self.next_action_values,
                                                                    feed_dict={self.next_observations: next_observations})

        action_values_target = rewards + self.discount_rate * (1.0 - dones) * np.max(action_values_target_network, 1)
        _, td_error = self.session.run([self.optimize, self.td_errors],
                                           feed_dict={
                                               self.observations: observations,
                                               self.actions: actions,
                                               self.action_values_target: action_values_target
                                           })
        return td_error


    def build(self):
        self.observations = tf.placeholder(tf.uint8, [None], name="observations")
        self.next_observations = tf.placeholder(tf.uint8, [None], name="next_observations")

        self.actions = tf.placeholder(tf.uint8, [None], name="actions")
        self.action_values_target = tf.placeholder(tf.float32, [None], name="action_values_target")

        self.action_values = action_value_function(inputs=tf.one_hot(self.observations,16),
                                                   action_space=self.action_space,
                                                   scope="network")

        self.next_action_values = action_value_function(inputs=tf.one_hot(self.next_observations,16),
                                                        action_space=self.action_space,
                                                        scope="target_network")

        self.action_value_variables = variables_with_scope("network")
        self.action_value_target_variables = variables_with_scope("target_network")

        selected_action_values = tf.reduce_sum(self.action_values * tf.one_hot(self.actions, self.action_space), 1)
        self.td_errors = selected_action_values - self.action_values_target
        losses = huber_loss(self.td_errors)
        loss = tf.reduce_mean(losses)

        if self.clip_norm is not None:
            gradients = self.optimizer.compute_gradients(loss)
            for i, (gradient, variable) in enumerate(gradients):
                if gradient is not None:
                    gradients[i] = (tf.clip_by_norm(t=gradient, clip_norm=self.clip_norm), variable)
            self.optimize = self.optimizer.apply_gradients(gradients)
        else:
            self.optimize = self.optimizer.minimize(loss)

        self.best_action = tf.argmax(self.action_values, axis=1)[0]