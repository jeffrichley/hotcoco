import gym
from HotEnv import HotEnv
import cv2

import time

# env = gym.make('CartPole-v0')
env = HotEnv()
for i_episode in range(20000000):

    if i_episode % 1000 == 0:
        print(i_episode)

    observation = env.reset()
    done = False
    # for t in range(100):
    while not done:
        # pic = env.render()

        # print(observation)
        shark_1_action = env.action_space.sample()
        shark_2_action = env.action_space.sample()
        # observation, reward, done, info = env.step((4, 3))
        observation, reward, done, info = env.step((shark_1_action, shark_2_action))

        # if reward == 100:
        #     pic = env.render(mode='rgb_array')
        #     cv2.imwrite('winner.png', cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))

        # time.sleep(.05)
        # time.sleep(.5)

        if done:
            observation = env.reset()
            break

env.close()
