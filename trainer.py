import ray
from ray.util.queue import Queue
from environment_runner import play_knights_and_zombies
from cleaning_prepping import clean_data
from learning import train_learners


def train():
    trainer_names = ['archer_0', 'archer_1', 'knight_0', 'knight_1']
    weight_distribution_frequency = 20

    # number_of_concurrent_games = 1
    number_of_concurrent_games = 2
    # batch_size = 512
    batch_size = 128
    # batch_size = 16

    input_size = 135
    num_agents = 4
    num_agent_actions = 6
    num_joint_actions = num_agent_actions**num_agents

    # create the communication queues
    play_queue = Queue(maxsize=100)
    cleaned_queue = Queue(maxsize=100)
    weight_update_queues = [Queue(maxsize=100) for _ in range(number_of_concurrent_games)]

    # setup the playing of games, these will play asynchronously
    play_game_ref = play_knights_and_zombies.remote(trainer_names=trainer_names,
                                                    input_size=input_size,
                                                    num_joint_actions=num_joint_actions,
                                                    play_queue=play_queue,
                                                    number_of_concurrent_games=number_of_concurrent_games,
                                                    weight_update_queues=weight_update_queues)

    # cleaning will happen on the fly when a game finishes
    clean_data.remote(play_queue=play_queue, cleaned_queue=cleaned_queue)

    # train with the received data
    train_learners.remote(training_queue=cleaned_queue,
                          input_size=input_size,
                          num_joint_actions=num_joint_actions,
                          num_agent_actions=num_agent_actions,
                          batch_size=batch_size,
                          weight_distribution_frequency=weight_distribution_frequency,
                          weight_update_queues=weight_update_queues)

    # just keep on playing
    ray.get(play_game_ref)


if __name__ == '__main__':
    train()


