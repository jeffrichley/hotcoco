import ray
from pettingzoo.butterfly import knights_archers_zombies_v10


class ZooRunner():

    # def __init__(self, agent_num, play_queue):
    def __init__(self, agent_num, play_queue):
        self.agent_num = agent_num
        self.play_queue = play_queue
        self.env = self.create_env()

    def create_env(self):
        raise Exception('Base class is not instantiable')

    def run_game(self):
        self.env.reset()
        all_observations = {}
        previous_states = {}
        previous_actions = {}
        for agent in self.env.agent_iter():

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

                action = self.get_action(next_state, agent)
                previous_actions[agent] = action

            self.env.step(action)

        self.play_queue.put(all_observations)

        return self.agent_num

    def get_action(self, observation, agent):
        action = self.env.action_space(agent).sample()
        return action



@ray.remote
class KnightsZombiesRunner(ZooRunner):

    def __init__(self, agent_num, play_queue):
        super().__init__(agent_num=agent_num, play_queue=play_queue)

    def create_env(self):
        # return knights_archers_zombies_v10.env(render_mode='human')
        return knights_archers_zombies_v10.env()


@ray.remote
def play_knights_and_zombies(play_queue, number_of_concurrent_games=4):
    runners = [KnightsZombiesRunner.remote(agent_num=agent_num, play_queue=play_queue) for agent_num in range(number_of_concurrent_games)]
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

    read_queue.remote(play_queue=play_queue)
    play_ref = play_knights_and_zombies.remote(play_queue=play_queue)
    print(ray.get(play_ref))
    print('done playing')



