import tensorflow as tf

class Model():
    def __init__(self, observation_space, action_space):
        self.inputs = tf.placeholder(shape=[1], dtype=tf.int32)
        self.q_out = tf.layers.dense(inputs=tf.one_hot(self.inputs, observation_space), units=action_space, activation=None)

        self.predict_action = tf.argmax(self.q_out, 1)
        self.next_q = tf.placeholder(shape=[1, action_space], dtype=tf.float32)

        loss = tf.reduce_sum(tf.squared_difference(self.next_q, self.q_out))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        self.optimize = optimizer.minimize(loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        

    def predict_q(self, observation):
        actions, q = self.session.run([self.predict_action, self.q_out], feed_dict={self.inputs: [observation]})
        return actions[0], q

    def predict_q_next(self, observation):
        return self.session.run(self.q_out, feed_dict={self.inputs: [observation]})

    def train(self, observation, target_q):
        self.session.run(self.optimize, feed_dict={self.inputs: [observation], self.next_q: target_q})