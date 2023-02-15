import math
import matplotlib.pyplot as plt

min_epsilon = 0.1
max_epsilon = 1.0
epsilon_decay = 0.001

epsilons = []
min_step = None

for number_of_steps_played in range(40000):
    value = max(min_epsilon, min_epsilon + (max_epsilon - min_epsilon) * math.exp(-epsilon_decay * number_of_steps_played))
    epsilons.append(value)

    # print(number_of_steps_played, value)
    if min_step is None and value <= 0.105:
        min_step = number_of_steps_played
        print(min_step)

plt.plot(epsilons)
# plt.title('Effect of Learning Rate')
# plt.xlabel('Episodes')
# plt.ylabel('Rolling Avg 100 Episode Reward')

plt.show()
