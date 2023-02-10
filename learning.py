import ray
import numpy as np
from tensorflow import keras
from keras import layers

from coco_utils import compute_coco_distributed


class Memory:

    def __init__(self, memory_size=1000000):
        # how many training samples should we remember?
        self.memory_size = memory_size

        # the actual memory items
        self.states = None
        self.actions = None
        self.rewards = None
        self.state_primes = None
        self.coco_cache = None

        self.memory_count = -1

    def clear_memory(self):
        self.states = None
        self.actions = None
        self.rewards = None
        self.state_primes = None
        self.coco_cache = None

        self.memory_count = -1

    def remember(self, state, action, reward, state_prime):

        if self.states is None:
            self.states = np.empty((self.memory_size, state.shape[0]))
            self.actions = np.empty(self.memory_size)
            self.rewards = np.empty((self.memory_size))
            self.state_primes = np.empty((self.memory_size, state_prime.shape[0]))
            self.coco_cache = np.full(self.memory_size, np.nan)

        self.memory_count += 1

        # remember everything we've been given
        idx = self.memory_count % self.memory_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_primes[idx] = state_prime

    def sample_memory(self, count, sample_idx=None):

        # randomly sample from the memory
        if sample_idx is None:
            sample_idx = np.random.randint(low=0, high=min(self.memory_count, self.memory_size), size=count)

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        rewards = self.rewards[sample_idx]
        state_primes = self.state_primes[sample_idx]
        coco_values = self.coco_cache[sample_idx]
        # print('***', coco_values.shape)

        return sample_idx, states, actions, rewards, state_primes, coco_values

    def save_coco_values(self, idxs, coco_values):
        self.coco_cache[idxs] = coco_values

    def num_samples(self):
        # how many samples do we have?
        return min(self.memory_count, self.memory_size)


@ray.remote
class Trainer:

    def __init__(self, trainer_names, input_size,
                 num_joint_actions, num_agent_actions,
                 gamma=0.99, batch_size=512,
                 num_coco_calculation_splits=4):
        self.trainer_names = trainer_names
        self.input_size = input_size
        self.num_joint_actions = num_joint_actions
        self.num_players = len(self.trainer_names)
        self.player_num_actions = np.full(self.num_players, num_agent_actions)
        self.player_powers = np.array(range(self.num_players-1, -1, -1))  # used for indexing joint actions
        self.joint_action_index_multiplier = np.tile(np.power(self.player_num_actions, self.player_powers), (batch_size, 1)).transpose()

        self.num_coco_calculation_splits = num_coco_calculation_splits

        # learning bits
        self.gamma = 0.99
        self.batch_size = batch_size
        self.optimizer = keras.optimizers.Adam()
        self.loss_function = keras.losses.Huber()

        # actual neural nets to update
        self.models = {}
        self.target_models = {}
        self.memories = {}
        for name in trainer_names:
            self.models[name] = self.create_q_model()
            self.target_models[name] = self.create_q_model()
            self.memories[name] = Memory(memory_size=100000)

    def create_q_model(self):

        # create the networks
        inputs_vectors = layers.Input(shape=self.input_size)

        # dense policy layers
        dense_layer0 = layers.Dense(128, activation='swish')(inputs_vectors)
        dense_layer1 = layers.Dense(128, activation='swish')(dense_layer0)
        dense_layer2 = layers.Dense(64, activation='swish')(dense_layer1)

        # policy output
        policy_output = layers.Dense(self.num_joint_actions, activation=None)(dense_layer2)

        model = keras.Model(inputs=inputs_vectors, outputs=policy_output)
        model.compile(optimizer=self.optimizer, loss=self.loss_function)

        return model

    def add_data(self, new_training_data):
        for name in self.trainer_names:
            memory = self.memories[name]
            for state, action, reward, state_prime in new_training_data[name]:
                memory.remember(state=state, action=action, reward=reward, state_prime=state_prime)

    def train_nn(self):
        # create the arrays to store the samples
        all_states = np.zeros((self.num_players, self.batch_size, self.input_size))
        all_actions = np.zeros((self.num_players, self.batch_size))
        all_rewards = np.zeros((self.num_players, self.batch_size))

        # used for calculating coco values
        # TODO: switch batch size and num players to be consistent
        all_future_rewards = np.zeros((self.batch_size, self.num_players, self.num_joint_actions))
        all_cached_coco_values = np.full((self.batch_size, self.num_players), np.nan)

        # 1. pull from the replay buffers all the information needed to do a learning step
        sample_idx = None
        for idx, name in enumerate(self.trainer_names):
            sample_idx, states, actions, rewards, state_primes, cached_coco_values = self.memories[name].sample_memory(self.batch_size, sample_idx=sample_idx)
            all_states[idx] = states
            all_actions[idx] = actions
            all_rewards[idx] = rewards
            # print(all_cached_coco_values.shape, all_cached_coco_values[idx].shape, cached_coco_values.shape)
            all_cached_coco_values[:, idx] = cached_coco_values

            # 2. predict what the payoff matrices are
            future_reward = self.target_models[name](state_primes, training=False)
            all_future_rewards[:, idx] = future_reward

            # 3. correct all of the terminal states to be all
            # --- don't think we will need to do this for now

        # 4. calculate the coco values for each player
        # need to clean any nans from the coco calculations
        # coco_values = np.zeros(rewards.shape)
        # TODO: need to cache coco results
        # TODO: need to check for bad values i.e. None
        coco_values = compute_coco_distributed(all_future_rewards, 8, self.num_players, all_cached_coco_values)

        # cache the coco values, they are expensive!
        for idx, name in enumerate(self.trainer_names):
            self.memories[name].save_coco_values(sample_idx, coco_values[:, idx])

        # print((coco_values == None).any())

        # 5. calculate the Q-values to be learned
        # updated_q_value = rewards + self.gamma * coco_values


        # 6. calculate the index of the actual joint action that was played
        joint_actions = (all_actions * self.joint_action_index_multiplier).sum(axis=0)
        # print(self.joint_action_index_multiplier, joint_actions, all_actions)

        # 7. create the masks that will be used to make sure we only learn from the actions we took

        # 8. actually perform the update



        # TODO: periodically need to swap brains

        print('done with training')


@ray.remote
def train_learners(training_queue, input_size, num_joint_actions, num_agent_actions):
    trainer_ref = None

    while True:
        # empty out the data queue before we go back to training
        new_data_received = False
        while not training_queue.empty():
            new_data_received = True
            all_training_data = training_queue.get()

            # make sure the trainer is actually created
            if trainer_ref is None:
                trainer_ref = Trainer.remote(trainer_names=all_training_data.keys(),
                                             input_size=input_size,
                                             num_joint_actions=num_joint_actions,
                                             num_agent_actions=num_agent_actions)

            trainer_ref.add_data.remote(all_training_data)

        if new_data_received:
            trainer_ref.train_nn.remote()

        # TODO: periodically test the policies
