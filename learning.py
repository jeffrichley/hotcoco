import time
import ray
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from coco_utils import compute_coco_distributed
from learner_base import BaseLearner


class Memory:

    def __init__(self, num_players, memory_size=1000000):
        # how many training samples should we remember?
        self.memory_size = memory_size

        self.num_players = num_players

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
            self.states = np.empty((self.memory_size, self.num_players, state.shape[1]))
            self.actions = np.empty((self.memory_size, self.num_players))
            self.rewards = np.empty((self.memory_size, self.num_players))
            self.state_primes = np.empty((self.memory_size, self.num_players, state_prime.shape[1]))
            self.coco_cache = np.full((self.memory_size, self.num_players), np.nan)

        self.memory_count += 1

        # remember everything we've been given
        idx = self.memory_count % self.memory_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_primes[idx] = state_prime

    def sample_memory(self, count):

        # randomly sample from the memory
        sample_idx = np.random.randint(low=0, high=min(self.memory_count, self.memory_size), size=count)

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        rewards = self.rewards[sample_idx]
        state_primes = self.state_primes[sample_idx]
        coco_values = self.coco_cache[sample_idx]

        return sample_idx, states, actions, rewards, state_primes, coco_values

    def save_coco_values(self, idxs, coco_values):
        self.coco_cache[idxs] = coco_values

    def num_samples(self):
        # how many samples do we have?
        return min(self.memory_count, self.memory_size)


@ray.remote
class Trainer(BaseLearner):

    def __init__(self, trainer_names, input_size,
                 num_joint_actions, num_agent_actions,
                 gamma=0.99, batch_size=512,
                 num_coco_calculation_splits=4):

        super().__init__(trainer_names=trainer_names, input_size=input_size, num_joint_actions=num_joint_actions)

        self.trainer_names = trainer_names
        self.input_size = input_size
        self.num_joint_actions = num_joint_actions
        self.num_players = len(self.trainer_names)
        self.player_num_actions = np.full(self.num_players, num_agent_actions)
        self.player_powers = np.array(range(self.num_players-1, -1, -1))  # used for indexing joint actions
        self.joint_action_index_multiplier = np.tile(np.power(self.player_num_actions, self.player_powers), (batch_size, 1))

        self.num_coco_calculation_splits = num_coco_calculation_splits

        # learning bits
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = Memory(num_players=self.num_players, memory_size=100000)

    def add_data(self, new_training_data):
        states = None
        actions = None
        rewards = None
        state_primes = None
        for idx, name in enumerate(self.trainer_names):
            # memory = self.memories[name]
            for state, action, reward, state_prime in new_training_data[name]:
                if states is None:
                    states = np.zeros((self.num_players, state.shape[0]))
                    actions = np.zeros(self.num_players)
                    rewards = np.zeros(self.num_players)
                    state_primes = np.zeros((self.num_players, state.shape[0]))

                states[idx] = state
                actions[idx] = action
                rewards[idx] = reward
                state_primes[idx] = state_prime

                self.memory.remember(state=states, action=actions, reward=rewards, state_prime=state_primes)

        self.memory.remember(state=states, action=actions, reward=rewards, state_prime=state_primes)

    def train_nn(self):
        # used for calculating coco values
        # TODO: switch batch size and num players to be consistent
        all_future_rewards = np.zeros((self.batch_size, self.num_players, self.num_joint_actions))

        # 1. pull from the replay buffers all the information needed to do a learning step
        sample_idx, states, actions, rewards, state_primes, cached_coco_values = self.memory.sample_memory(self.batch_size)

        # 2. predict what the payoff matrices are
        for idx, name in enumerate(self.trainer_names):
            # TODO: Can we do this in one shot?
            future_reward = self.target_models[name](state_primes[:, idx], training=False)
            all_future_rewards[:, idx] = future_reward

        # 3. correct all of the terminal states to be all
        # --- don't think we will need to do this for now

        # 4. calculate the coco values for each player
        # need to clean any nans from the coco calculations
        # TODO: need to check for bad values i.e. None
        coco_values = compute_coco_distributed(all_future_rewards, 8, self.num_players, cached_coco_values)

        # cache the coco values, they are expensive!
        self.memory.save_coco_values(sample_idx, coco_values)

        # 5. calculate the Q-values to be learned
        updated_q_values = rewards + self.gamma * coco_values


        # 6. calculate the index of the actual joint action that was played
        joint_actions = joint_actions = (actions * self.joint_action_index_multiplier).sum(axis=1)
        # print(actions.shape, self.joint_action_index_multiplier.shape, joint_actions.shape)
        # print('actions')
        # print(actions)
        # print('multiplier')
        # print(self.joint_action_index_multiplier)
        # print(self.joint_action_index_multiplier, joint_actions, actions)

        # 7. create the masks that will be used to make sure we only learn from the actions we took
        masks = tf.one_hot(joint_actions, self.num_joint_actions)

        # 8. actually perform the update
        self.update(states, masks, updated_q_values)

        # TODO: periodically need to swap brains
        # print('done with training')

    def update(self, all_state_sample, all_masks, all_updated_q_values):  # , learner_name=None, epoch=0):
        masks = tf.convert_to_tensor(all_masks)

        for idx, name in enumerate(self.trainer_names):
            # print(f'**** start {name}')
        # for idx in range(2):
            name = self.trainer_names[idx]
            model = self.models[name]

            state_sample = all_state_sample[:, idx]
            # print('state sample', all_state_sample.shape, state_sample.shape)
            state_sample = tf.convert_to_tensor(state_sample)

            updated_q_values = all_updated_q_values[:, idx]
            # print('updated_q_values', all_updated_q_values.shape, updated_q_values.shape)
            updated_q_values = tf.convert_to_tensor(updated_q_values, dtype=tf.float32)

            # update the main model
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                # by first predicting the q values
                q_values = model(state_sample)

                # print(q_values.shape, masks.shape)
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

                # print(q_values.shape, q_action.shape, updated_q_values.shape)

                # Calculate loss between new Q-value and old Q-value
                # can use sample_weight to apply individual loss scaling
                # loss = self.loss_function(updated_q_values, q_action)
                loss = self.loss_functions[name](updated_q_values, q_action)

            # calculate and apply the gradients to the model
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 2) for g in grads]
            # self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            self.optimizers[name].apply_gradients(zip(grads, model.trainable_variables))

            # periodic tensorboard logging
            # if learner_name is not None and epoch % 10 == 0:
            #     tf.summary.scalar('loss_' + learner_name, loss, step=epoch)
            #     top = tf.abs(q_action - updated_q_values)
            #     run_error = top / tf.reduce_sum(top)
            #     avg_error = tf.math.reduce_mean(run_error)
            #     max_error = tf.math.reduce_max(top)
            #     tf.summary.scalar('avg_error_' + learner_name, avg_error, step=epoch)
            #     tf.summary.scalar('max_error_' + learner_name, max_error, step=epoch)
            #
            #     max_norm = tf.reduce_max([tf.norm(grad) for grad in grads])
            #     tf.summary.scalar('max_gradient_norm_' + learner_name, max_norm, step=epoch)


@ray.remote
def train_learners(training_queue, input_size, num_joint_actions, num_agent_actions, batch_size):
    trainer_ref = None

    num_rounds = 0
    num_pulled = 0

    while True:
        # empty out the data queue before we go back to training
        new_data_received = False
        while not training_queue.empty():
            new_data_received = True
            all_training_data = training_queue.get()

            # make sure the trainer is actually created
            if trainer_ref is None:
                trainer_ref = Trainer.remote(trainer_names=list(all_training_data.keys()),
                                             input_size=input_size,
                                             num_joint_actions=num_joint_actions,
                                             num_agent_actions=num_agent_actions,
                                             batch_size=batch_size)

            trainer_ref.add_data.remote(all_training_data)

        if new_data_received:
            trainer_ref.train_nn.remote()
        else:
            time.sleep(0.01)

        # TODO: periodically test the policies
