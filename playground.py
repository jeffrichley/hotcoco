import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10

trainer_names = ['archer_0', 'archer_1', 'knight_0', 'knight_1']
env = knights_archers_zombies_v10.env()
env.reset()

observations = np.zeros((1, 4, 135))
for idx, name in enumerate(trainer_names):
    observation = np.reshape(env.observe(name), 135)
    observations[0, idx] = observation
    print(observation.shape)
    print('*********')

print(observations.shape)

for agent in env.agent_iter():
    next_state, reward, termination, truncation, info = env.last()
    if termination:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
