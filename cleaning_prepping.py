import ray
import numpy as np
import ray_utils


@ray.remote
def clean_data(play_queue, cleaned_queue):
    """
    Takes data from game play and creates usable data for training
    :param play_queue: The queue to read from for game trajectories
    :param cleaned_queue: The queue to publish cleaned data to
    """

    while True:
        # read data from the game trajectory queue
        data = play_queue.get(block=True)

        # clean each player's individual data
        all_cleaned = {}
        for key in data.keys():
            if key not in all_cleaned.keys():
                # lazy instantiation
                all_cleaned[key] = []

            player_data = data[key]
            player_cleaned = all_cleaned[key]
            for entry in player_data:
                state_list, action, reward, state_prime_list = entry

                # the cleaning is a simple concatenation of each of the players and monsters information
                state = np.concatenate(state_list)
                state_prime = np.concatenate(state_prime_list)

                player_cleaned.append((state, action, reward, state_prime))

        cleaned_queue.put(all_cleaned)
