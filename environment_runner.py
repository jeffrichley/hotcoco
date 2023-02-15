import numpy as np
from numpy import unravel_index
import ray
from pettingzoo.butterfly import knights_archers_zombies_v10

from learner_base import BaseLearner


class ZooRunner:

    # def __init__(self, agent_num, play_queue):
    def __init__(self, agent_num, play_queue, trainer_names, input_size, num_joint_actions):
        self.trainer_names = trainer_names
        self.num_joint_actions = num_joint_actions

        self.agent_num = agent_num
        self.number_of_players = len(trainer_names)
        self.play_queue = play_queue
        self.env = self.create_env()

        self.epsilon = 0.75

        self.learner = BaseLearner(trainer_names=trainer_names,
                                   input_size=input_size,
                                   num_joint_actions=num_joint_actions)

    def create_env(self):
        raise Exception('Base class is not instantiable')

    def run_game(self):
        self.env.reset()
        all_observations = {}
        previous_states = {}
        previous_actions = {}

        agent_actions = {}

        for agent in self.env.agent_iter():

            if agent not in agent_actions:
                print(agent, agent_actions)
                agent_actions = self.get_actions()

            if agent not in all_observations.keys():
                all_observations[agent] = []
                previous_states[agent] = None
                previous_actions[agent] = None

            next_state, reward, termination, truncation, info = self.env.last()

            if termination:
                action = None
            else:
                # because this agent is still playing, we need to capture the data for learning
                if previous_states[agent] is not None:
                    previous_state = previous_states[agent]
                    previous_action = previous_actions[agent]
                    all_observations[agent].append((previous_state, previous_action, reward, next_state))

                previous_states[agent] = next_state

                action = agent_actions[agent]

                previous_actions[agent] = action

            agent_actions.pop(agent, None)
            self.env.step(action)

        self.play_queue.put(all_observations)

        # TODO: need to reduce epsilon

        return self.agent_num

    def get_actions(self):
        actions = {}
        if np.random.random() < self.epsilon:
            for agent_name in self.trainer_names:
                actions[agent_name] = self.env.action_space(agent_name).sample()
        else:
            # TODO: need to actually use a policy from time to time
            predictions = np.zeros((self.number_of_players, self.num_joint_actions))
            for idx, name in enumerate(self.trainer_names):
                observation = np.reshape(self.env.observe(name), (1, 135))

                prediction = self.learner.query_model(name, observation, training=False)
                predictions[idx] = prediction

            predictions = predictions.sum(axis=0)
            predictions = np.reshape(predictions, [6] * self.number_of_players)
            calculated_actions = unravel_index(predictions.argmax(), predictions.shape)

            for agent_name, predicted_action in zip(self.trainer_names, calculated_actions):
                actions[agent_name] = predicted_action

        return actions


@ray.remote
class KnightsZombiesRunner(ZooRunner):

    def __init__(self, agent_num, play_queue, trainer_names, input_size, num_joint_actions):
        super().__init__(agent_num=agent_num, play_queue=play_queue,
                         trainer_names=trainer_names, input_size=input_size,
                         num_joint_actions=num_joint_actions)

    def create_env(self):
        return knights_archers_zombies_v10.env(render_mode='human')
        # return knights_archers_zombies_v10.env()


@ray.remote
def play_knights_and_zombies(trainer_names, input_size, num_joint_actions, play_queue, number_of_concurrent_games=1):
    runners = [KnightsZombiesRunner.remote(agent_num=agent_num,
                                           play_queue=play_queue,
                                           trainer_names=trainer_names,
                                           input_size=input_size,
                                           num_joint_actions=num_joint_actions) for agent_num in range(number_of_concurrent_games)]
    not_done = [runner.run_game.remote() for runner in runners]
    while not_done:
        done, not_done = ray.wait(not_done)
        agent_num = ray.get(done)[0]

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



