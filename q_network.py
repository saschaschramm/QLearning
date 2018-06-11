import tensorflow as tf

def action_value_function(inputs, observation_space, action_space, scope):
    with tf.variable_scope(scope):
        return tf.layers.dense(inputs=tf.one_hot(inputs, observation_space),
                                                 units=action_space,
                                                 activation=None)

def variables_with_scope(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

def update_target_network_operations():
    variables_network = variables_with_scope("network")
    variables_target_network = variables_with_scope("target_network")
    assign_operations = []
    for variable, variable_target in zip(sorted(variables_network, key=lambda v: v.name),
                                         sorted(variables_target_network, key=lambda v: v.name)):
        assign_operations.append(variable_target.assign(variable))
    return assign_operations

class QNetwork():
    def __init__(self, observation_space, action_space):
        self.inputs = tf.placeholder(shape=[1], dtype=tf.int32)

        self.action_values_network = action_value_function(inputs=self.inputs,
                                                           observation_space=observation_space,
                                                           action_space=action_space,
                                                           scope="network")

        self.action_values_target_network = action_value_function(inputs=self.inputs,
                                                           observation_space=observation_space,
                                                           action_space=action_space,
                                                           scope="target_network")

        self.update_target_network = update_target_network_operations()

        self.actions = tf.argmax(self.action_values_network, axis=1)
        self.target_action_values = tf.placeholder(shape=[1, action_space], dtype=tf.float32)
        loss = tf.reduce_sum(tf.squared_difference(self.target_action_values, self.action_values_network))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.optimize = optimizer.minimize(loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def predict_action(self, observation): # train every iteration
        actions, action_values = self.session.run([self.actions, self.action_values_network],
                                                  feed_dict={self.inputs: [observation]})
        return actions[0], action_values[0]

    def predict_action_value(self, observation): # only update wit certain frequency
        return self.session.run(self.action_values_target_network,
                                feed_dict={self.inputs: [observation]})

    def train(self, observation, target_action_values):
        self.session.run(self.optimize, feed_dict={self.inputs: [observation],
                                                   self.target_action_values: target_action_values})

    def update(self):
        self.session.run(self.update_target_network)