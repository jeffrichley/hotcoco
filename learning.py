import time
import ray
import numpy as np
import tensorflow as tf

from coco_utils import compute_coco_distributed
from learner_base import BaseLearner


class Memory:
    """
    Responsible for remembering and forgetting data samples
    """

    def __init__(self, num_players, memory_size=100000):
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
        """
        Throws away all stored information
        """
        self.states = None
        self.actions = None
        self.rewards = None
        self.state_primes = None
        self.coco_cache = None

        self.memory_count = -1

    def remember(self, state, action, reward, state_prime):

        # lazy instantiation
        if self.states is None:
            self.states = np.empty((self.memory_size, self.num_players, state.shape[1]))
            self.actions = np.empty((self.memory_size, self.num_players))
            self.rewards = np.empty((self.memory_size, self.num_players))
            self.state_primes = np.empty((self.memory_size, self.num_players, state_prime.shape[1]))
            self.coco_cache = np.full((self.memory_size, self.num_players), np.nan)

        # how many samples have we been given
        self.memory_count += 1

        # remember everything we've been given
        idx = self.memory_count % self.memory_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_primes[idx] = state_prime

    def sample_memory(self, count):
        """
        Get a number of samples of training data
        :param count: How many samples of training data to retrieve
        :return: The requested training data
        """

        # randomly sample from the memory
        sample_idx = np.random.randint(low=0, high=min(self.memory_count, self.memory_size), size=count)

        # pull the information out of the memory to return
        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        rewards = self.rewards[sample_idx]
        state_primes = self.state_primes[sample_idx]
        coco_values = self.coco_cache[sample_idx]

        return sample_idx, states, actions, rewards, state_primes, coco_values

    def save_coco_values(self, idxs, coco_values):
        """
        We need to cache computed coco values, they are very expensive to calculate
        :param idxs: What indexes were sampled from so we can update the cache
        :param coco_values: The computed coco values to be cached
        """
        self.coco_cache[idxs] = coco_values

    def num_samples(self):
        """
        How many samples do we have?
        :return: The number of samples we have to pull from
        """
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

        # self.memory.remember(state=states, action=actions, reward=rewards, state_prime=state_primes)

    def train_nn(self, weight_update_queues=None):
        # used for calculating coco values
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
        coco_values = compute_coco_distributed(all_future_rewards, 12, self.num_players, cached_coco_values)

        # cache the coco values, they are expensive!
        self.memory.save_coco_values(sample_idx, coco_values)

        # 5. calculate the Q-values to be learned
        updated_q_values = rewards + self.gamma * coco_values


        # 6. calculate the index of the actual joint action that was played
        joint_actions = (actions * self.joint_action_index_multiplier).sum(axis=1)

        # 7. create the masks that will be used to make sure we only learn from the actions we took
        masks = tf.one_hot(joint_actions, self.num_joint_actions)

        # 8. actually perform the update
        self.update(states, masks, updated_q_values)

        if weight_update_queues is not None:
            new_weights = {}
            for agent_name in self.trainer_names:
                new_weights[agent_name] = self.models[agent_name].get_weights()

            for queue in weight_update_queues:
                queue.put(new_weights)

        # TODO: periodically need to swap brains

    def update(self, all_state_sample, all_masks, all_updated_q_values):
        masks = tf.convert_to_tensor(all_masks)

        # we need to do an update for each agent's neural network
        for idx, name in enumerate(self.trainer_names):
            name = self.trainer_names[idx]
            model = self.models[name]

            state_sample = all_state_sample[:, idx]
            state_sample = tf.convert_to_tensor(state_sample)

            updated_q_values = all_updated_q_values[:, idx]
            updated_q_values = tf.convert_to_tensor(updated_q_values, dtype=tf.float32)

            # update the main model
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                # by first predicting the q values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

                # Calculate loss between new Q-value and old Q-value
                # can use sample_weight to apply individual loss scaling
                loss = self.loss_functions[name](updated_q_values, q_action)

            # calculate and apply the gradients to the model
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 2) for g in grads]
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
def train_learners(training_queue, input_size, num_joint_actions, num_agent_actions, batch_size,
                   weight_distribution_frequency, weight_update_queues):
    """
    Remote method that coordinates all of the gathering and saving of trajectory information as well as
    the actual training of the neural networks.
    :param training_queue: The queue to pull data from
    :param input_size: The size of the inputs into the neural networks
    :param num_joint_actions: The number of joint actions possible
    :param num_agent_actions: The number of actions agents can take
    :param batch_size: The size of the learning batches
    :param weight_distribution_frequency: How often should we update weights?
    :param weight_update_queues: The queues to push new model weights through
    """
    trainer_ref = None
    max_number_of_no_data_training = 5
    number_of_no_data_training = 0
    data_trains = 0
    no_data_trains = 0
    num_training_iterations = 0

    while True:
        # empty out the data queue before we go back to training
        new_data_received = False

        pulls = 0
        num_data = 0

        while not training_queue.empty():
            pulls += 1
            new_data_received = True
            all_training_data = training_queue.get()

            # make sure the trainer is actually created
            if trainer_ref is None:
                trainer_ref = Trainer.remote(trainer_names=list(all_training_data.keys()),
                                             input_size=input_size,
                                             num_joint_actions=num_joint_actions,
                                             num_agent_actions=num_agent_actions,
                                             batch_size=batch_size,
                                             gamma=0.99,
                                             num_coco_calculation_splits=8)

            # save the data received
            trainer_ref.add_data.remote(all_training_data)

            # print(all_training_data)
            num_data += len(all_training_data['archer_0'])
            # for something in all_training_data[name]:
            #     num_data += 1

        if new_data_received:
            # if we received new data we need to do a training loop
            start = time.time()
            # change this to only periodically updating weights
            if num_training_iterations % weight_distribution_frequency == 0:
                ray.get(trainer_ref.train_nn.remote(weight_update_queues=weight_update_queues))
            else:
                ray.get(trainer_ref.train_nn.remote(weight_update_queues=None))

            end = time.time()
            print('training data', pulls, num_data, end - start)
            data_trains += 1
            number_of_no_data_training = 0
            num_training_iterations += 1
        else:
            # slow down just a tiny bit if we didn't receive any new data
            if trainer_ref is not None and number_of_no_data_training < max_number_of_no_data_training:
                # print('no data training')
                # trainer_ref.train_nn.remote()
                no_data_trains += 1
                number_of_no_data_training += 1
            else:
                time.sleep(0.01)

        # if data_trains + no_data_trains > 0:
        #     print(data_trains, no_data_trains, data_trains / (data_trains + no_data_trains))



        # TODO: periodically test the policies
