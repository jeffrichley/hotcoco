import numpy as np

values = np.array([i for i in range(3**4)]).reshape((3, 3, 3, 3))
unravelled = np.array([i for i in range(3**4)])

print(values)
print(values.shape)
print(unravelled)
print(unravelled.shape)

all_num_actions = 3
num_agents = 4
num_actions = np.full(num_agents, all_num_actions)
# powers = np.array([2, 1, 0])
powers = np.array(range(num_agents-1, -1, -1))

idx = -1
total_actions = np.zeros((4, 3**4))
for p1_action in range(3):
    for p2_action in range(3):
        for p3_action in range(3):
            for p4_action in range(3):
                idx += 1
                total_actions[:, idx] = [p1_action, p2_action, p3_action, p4_action]

                # unravelled_index = p1_action * num_actions**2 + p2_action * num_actions + p3_action
                player_values = np.array([p1_action, p2_action, p3_action, p4_action])
                unravelled_index = (player_values * np.power(num_actions, powers)).sum()

                print(p1_action, p2_action, p3_action, p4_action)
                print('indexing', values[(p1_action, p2_action, p3_action, p4_action)])
                print('unravelled', unravelled[unravelled_index])



# values = np.array([2, 2, 2])
# jt_action = values * np.power(num_actions, powers)
# print(jt_action)
# powers = np.tile(powers, (27, 1)).transpose()
# print('powers', powers)
# print(total_actions)
tmp = (total_actions * np.tile(np.power(num_actions, powers), (3**4, 1)).transpose()).sum(axis=0)
print('tmp.txt', tmp)

# t = np.zeros((5, 5, 5, 5))
# print(t.shape, t.size)
