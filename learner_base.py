import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


class BaseLearner:
    """
    The base class for learners.  Main components are the experience
    replays as well as target and primary neural network.
    """

    def __init__(self, trainer_names, input_size, num_joint_actions):

        # basic information
        self.trainer_names = trainer_names
        self.input_size = input_size
        self.num_joint_actions = num_joint_actions

        # actual neural nets to update
        self.models = {}
        self.target_models = {}
        self.memories = {}
        self.optimizers ={}
        self.loss_functions = {}

        # create the primary and target networks
        for name in trainer_names:
            self.optimizers[name] = keras.optimizers.Adam()
            self.loss_functions[name] = keras.losses.Huber()

            self.models[name] = self.create_q_model(self.optimizers[name], self.loss_functions[name])
            # TODO: shouldn't really be reusing the optimizer and loss functions
            self.target_models[name] = self.create_q_model(self.optimizers[name], self.loss_functions[name])

    def create_q_model(self, optimizer, loss_function):

        # create the networks
        inputs_vectors = layers.Input(shape=self.input_size)

        # dense policy layers
        dense_layer0 = layers.Dense(128, activation='swish')(inputs_vectors)
        dense_layer1 = layers.Dense(128, activation='swish')(dense_layer0)
        dense_layer2 = layers.Dense(64, activation='swish')(dense_layer1)

        # policy output
        policy_output = layers.Dense(self.num_joint_actions, activation=None)(dense_layer2)

        model = keras.Model(inputs=inputs_vectors, outputs=policy_output)
        model.compile(optimizer=optimizer, loss=loss_function)

        return model

    def query_model(self, agent_name, data, training=False):
        """
        Queries the named network for predictions
        :param agent_name: Which agent are we querying
        :param data: The observations to query for
        :param training: Is the network training or just a basic query?  False will execute much faster
        :return: The predicted values from the given observations.
        """
        sample = tf.convert_to_tensor(data)
        return np.array(self.target_models[agent_name](sample, training=training))
