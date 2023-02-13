import math
import time
import numpy as np

actions = np.array([[2., 1., 4., 0.],
                    [3., 5., 0., 0.],
                    [0., 4., 0., 0.],
                    [5., 1., 0., 0.],
                    [3., 0., 0., 0.],
                    [5., 1., 1., 1.],
                    [3., 0., 0., 0.],
                    [5., 5., 5., 5.]])

num_players = 4
player_num_actions = 6
batch_size = 8

player_powers = np.array(range(num_players-1, -1, -1))
joint_action_index_multiplier = np.tile(np.power(player_num_actions, player_powers), (batch_size, 1))

final = joint_actions = (actions * joint_action_index_multiplier).sum(axis=1)

print('actions')
print(actions)
print('multiplier')
print(joint_action_index_multiplier)
print('final')
print(final)
