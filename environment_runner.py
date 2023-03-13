import math
import numpy as np
from numpy import unravel_index
import ray
from pettingzoo.butterfly import knights_archers_zombies_v10

from learner_base import BaseLearner


class ZooRunner:

    # def __init__(self, agent_num, play_queue):
    def __init__(self, agent_num, play_queue, trainer_names, input_size, num_joint_actions, weight_update_queue):

        # general information
        self.trainer_names = trainer_names
        self.num_joint_actions = num_joint_actions
        self.number_of_steps_played = 0

        # game information
        self.agent_num = agent_num
        self.number_of_players = len(trainer_names)
        self.play_queue = play_queue
        self.weight_update_queue = weight_update_queue
        self.env = self.create_env()

        # epsilon greedy sampling information
        self.epsilon = 0.99
        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.001

        # used for deciding actions
        self.learner = BaseLearner(trainer_names=trainer_names,
                                   input_size=input_size,
                                   num_joint_actions=num_joint_actions)

    def create_env(self):
        """
        Used to create a specialized environment.  Subclass and override this method
        :return: An environment created and preconfigured
        """
        raise Exception('Base class is not instantiable')

    def decay_epsilon(self):
        """
        Decay the epsilon greedy action probability
        """
        self.epsilon = max(self.min_epsilon, self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.epsilon_decay * self.number_of_steps_played))

    def check_model_updates(self):
        try:
            # we need to check to see if there are new weights to update our neural network with
            new_weights = self.weight_update_queue.get(block=False)
            self.learner.update_model(new_weights)
        except ray.util.queue.Empty:
            # there is a good chance there will be no update which will cause an ray.util.queue.Empty error
            # ew, using catching errors as control flow...sort of :(
            pass

    def run_game(self):
        """
        Responsible for actually running a game until completion.  Once the game is complete, the new trajectory
        information will be pushed out through the queue.
        :return: The number of the agent that is completing the game
        """

        # we need to check if there is an update to the weights of our model
        self.check_model_updates()

        self.env.reset()
        all_observations = {}
        previous_states = {}
        previous_actions = {}
        agent_actions = {}

        # iterate through each agent that is still alive in the environment
        for agent in self.env.agent_iter():

            # if the agent's name is not in the agent actions dictionary, it
            # means we have used all of the actions and are completing a new round of activity
            if agent not in agent_actions:
                agent_actions = self.get_actions()
                self.number_of_steps_played += 1
                self.decay_epsilon()

            if agent not in all_observations.keys():
                # lazy initialization
                all_observations[agent] = []
                previous_states[agent] = None
                previous_actions[agent] = None

            # get the information that happened since the last action for this agent
            next_state, reward, termination, truncation, info = self.env.last()

            if termination:
                # if we are here, this agent is dead
                action = None
            else:
                # Because this agent is still playing, we need to capture the data for learning.
                # It is important to remember we are needing to save the data from the
                # the previous round and then save this round's information for next rounds memory.
                if previous_states[agent] is not None:
                    previous_state = previous_states[agent]
                    previous_action = previous_actions[agent]
                    all_observations[agent].append((previous_state, previous_action, reward, next_state))

                previous_states[agent] = next_state
                action = agent_actions[agent]
                previous_actions[agent] = action

            # We are popping the agent's action so we can know when a round is complete.  Once
            # the agents are not in the action map, we will create a new action map.
            agent_actions.pop(agent, None)

            # the agent takes a step
            self.env.step(action)

        # post the agents' trajectories for processing by the next process
        self.play_queue.put(all_observations)

        return self.agent_num

    def get_actions(self):
        actions = {}

        if np.random.random() < self.epsilon:
            # we need to do some kind of valid random action
            for agent_name in self.trainer_names:
                actions[agent_name] = self.env.action_space(agent_name).sample()
        else:
            # COCO actions require all agents to take the highest valued joint action
            predictions = np.zeros((self.number_of_players, self.num_joint_actions))
            for idx, name in enumerate(self.trainer_names):
                observation = np.reshape(self.env.observe(name), (1, 135))
                prediction = self.learner.query_model(name, observation, training=False)
                predictions[idx] = prediction

            # take all the actions, sum them together, reshape, and get the argmax
            predictions = predictions.sum(axis=0)
            predictions = np.reshape(predictions, [6] * self.number_of_players)
            calculated_actions = unravel_index(predictions.argmax(), predictions.shape)

            # save the actions
            for agent_name, predicted_action in zip(self.trainer_names, calculated_actions):
                actions[agent_name] = predicted_action

        return actions


@ray.remote
class KnightsZombiesRunner(ZooRunner):

    def __init__(self, agent_num, play_queue, trainer_names, input_size, num_joint_actions, weight_update_queue):
        super().__init__(agent_num=agent_num, play_queue=play_queue,
                         trainer_names=trainer_names, input_size=input_size,
                         num_joint_actions=num_joint_actions,
                         weight_update_queue=weight_update_queue)

    def create_env(self):
        # return knights_archers_zombies_v10.env(render_mode='human')
        return knights_archers_zombies_v10.env()


@ray.remote
def play_knights_and_zombies(trainer_names, input_size, num_joint_actions, play_queue,
                             weight_update_queues, number_of_concurrent_games=1):
    # create a number of the game runners
    runners = [KnightsZombiesRunner.remote(agent_num=agent_num,
                                           play_queue=play_queue,
                                           trainer_names=trainer_names,
                                           input_size=input_size,
                                           num_joint_actions=num_joint_actions,
                                           weight_update_queue=weight_update_queue)
               for agent_num, weight_update_queue in zip(range(number_of_concurrent_games), weight_update_queues)]

    # kick off the playing of games
    not_done = [runner.run_game.remote() for runner in runners]
    while not_done:
        # wait for one game to complete
        done, not_done = ray.wait(not_done)
        # get which agent just completed
        agent_num = ray.get(done)[0]
        # get the game runner that just finished and start it again
        new_call = runners[agent_num].run_game.remote()
        not_done.append(new_call)


if __name__ == '__main__':

    @ray.remote
    def read_queue(play_queue):
        print('starting queue reader')
        while True:
            data = play_queue.get(block=True)
            print(data.keys())


    from ray.util.queue import Queue
    play_queue = Queue(maxsize=100)

    original_trainer_names = ['archer_0', 'archer_1', 'knight_0', 'knight_1']
    input_size = 135
    num_agents = 1
    num_players = 4
    num_agent_actions = 6
    num_joint_actions = num_agent_actions ** num_players

    read_queue.remote(play_queue=play_queue)
    play_ref = play_knights_and_zombies.remote(play_queue=play_queue,
                                               trainer_names=original_trainer_names,
                                               input_size=input_size,
                                               num_joint_actions=num_joint_actions)
    print(ray.get(play_ref))
    print('done playing')



