import ray
import numpy as np
import ray_utils


@ray.remote
def clean_data(play_queue, cleaned_queue):

    while True:
        data = play_queue.get(block=True)
        all_cleaned = {}
        for key in data.keys():
            if key not in all_cleaned.keys():
                all_cleaned[key] = []

            player_data = data[key]
            player_cleaned = all_cleaned[key]
            for entry in player_data:
                state_list, action, reward, state_prime_list = entry

                state = np.concatenate(state_list)
                state_prime = np.concatenate(state_prime_list)

                player_cleaned.append((state, action, reward, state_prime))

        cleaned_queue.put(all_cleaned)
