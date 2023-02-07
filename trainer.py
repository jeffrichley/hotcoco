import ray
from ray.util.queue import Queue
from environment_runner import play_knights_and_zombies
from cleaning_prepping import clean_data


def train():
    play_queue = Queue(maxsize=100)
    cleaned_queue = Queue(maxsize=100)

    # setup the playing of games, these will play asynchronously
    play_game_ref = play_knights_and_zombies.remote(play_queue=play_queue, number_of_concurrent_games=4)

    # cleaning will happen on the fly when a game finishes
    clean_data.remote(play_queue=play_queue, cleaned_queue=cleaned_queue)

    # just keep on playing
    ray.get(play_game_ref)


if __name__ == '__main__':
    train()


