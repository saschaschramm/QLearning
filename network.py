import tensorflow as tf

class Network():
    def __init__(self, observation_space, action_space):
        self.inputs = tf.placeholder(shape=[1], dtype=tf.int32)
        self.action_values = tf.layers.dense(inputs=tf.one_hot(self.inputs, observation_space), units=action_space, activation=None)
        self.actions = tf.argmax(self.action_values, axis=1)
        self.target_action_values = tf.placeholder(shape=[1, action_space], dtype=tf.float32)
        loss = tf.reduce_sum(tf.squared_difference(self.target_action_values, self.action_values))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.optimize = optimizer.minimize(loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def predict_action(self, observation):
        actions, q_values = self.session.run([self.actions, self.action_values], feed_dict={self.inputs: [observation]})
        return actions[0], q_values[0]

    def predict_action_value(self, observation):
        return self.session.run(self.action_values, feed_dict={self.inputs: [observation]})

    def train(self, observation, target_action_values):
        self.session.run(self.optimize, feed_dict={self.inputs: [observation], self.target_action_values: target_action_values})