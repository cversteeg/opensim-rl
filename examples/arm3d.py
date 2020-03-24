import os
from osim.env import Arm3DEnvMoBL
import pprint
import numpy as np

env = Arm3DEnvMoBL()
if __name__ == '__main__':
    observation = env.reset()
    print(env.osim_model.list_elements())
    for i in range(3000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        print(len(observation))
        if done:
            env.reset()
