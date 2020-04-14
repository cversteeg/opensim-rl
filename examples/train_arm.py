# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from osim.env.arm import Arm2DVecEnv
from osim.env import L2M2019Env

from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import argparse
import math


# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=500000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=True)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
args = parser.parse_args()

# set to get observation in array
#def _new_step(self, action, project=True, obs_as_dict=False):
#    return super(Arm2DEnv, self).step(action, project=project, obs_as_dict=obs_as_dict)
#Arm2DEnv.step = _new_step
# Load walking environment
env = Arm2DVecEnv(args.visualize)
# env = L2M2019Env(args.visualize)
env.reset()
#env.reset(verbose=True, logfile='arm_log.txt')

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

import tensorflow as tf

#env.reset(verbose=True, logfile='arm_log.txt')
tf.compat.v1.disable_eager_execution()

# Create networks for DDPG
# Next, we build a very simple model.
# observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
# flattened_observation = Flatten()(observation_input)
# x1 = Dense(32)(flattened_observation)
# x1 = Activation('relu')(x1)
# x1 = Dense(32)(x1)
# x1 = Activation('relu')(x1)
# x1 = Dense(32)(x1)
# x1 = Activation('relu')(x1)
# x1 = Dense(nb_actions)(x1)
# x1 = Activation('sigmoid')(x1)

# actor = Model(inputs = [observation_input], outputs=x1)

actor = Sequential()
actor.add(Flatten(input_shape= (1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64,name='critic_input')(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.005, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=1000, log_interval=10000)
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

if not args.train:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=20, visualize=True, nb_max_episode_steps=1000)