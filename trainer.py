import ray
from ray.util.queue import Queue
from environment_runner import play_knights_and_zombies
from cleaning_prepping import clean_data
from learning import train_learners


def train():
    input_size = 135
    num_agents = 4
    num_agent_actions = 6
    num_joint_actions = num_agent_actions**num_agents

    play_queue = Queue(maxsize=100)
    cleaned_queue = Queue(maxsize=100)

    # setup the playing of games, these will play asynchronously
    play_game_ref = play_knights_and_zombies.remote(play_queue=play_queue, number_of_concurrent_games=1)

    # cleaning will happen on the fly when a game finishes
    clean_data.remote(play_queue=play_queue, cleaned_queue=cleaned_queue)

    # train with the received data
    train_learners.remote(training_queue=cleaned_queue,
                          input_size=input_size,
                          num_joint_actions=num_joint_actions,
                          num_agent_actions=num_agent_actions)

    # just keep on playing
    ray.get(play_game_ref)


if __name__ == '__main__':
    train()


