# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:43:47 2020

@author: wrest
"""
import osim
import numpy as np
import sys
import argparse
import math

# Keras libraries 
from keras.optimizers import Adam
from keras.optimizers import RMSprop

# Keras RL
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Lambda, Cropping0D
from keras.utils.vis_utils import plot_model
import tensorflow as tf
def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func1(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func1)

def policy_nn(shape_in, shape_out, hidden_layers = 5, hidden_size = 64):
    total_input = Input(shape = (shape_in,), name = 'combined_input' )

    x = total_input
    for i in range(hidden_layers):
        x  = Dense(hidden_size)(x)
        x = Activation('relu')(x)
    x = Dense(shape_out)(x)
    x = Activation('sigmoid')(x)
    actor = Model(inputs= [total_input], outputs = x)
    return actor

# def policy_nn(shape_in, shape_out, hidden_layers = 3, hidden_size = 32):
    
#     total_input = Input(shape = (shape_in,), name = 'combined_input' )
#     muscle_input = Cropping0D(cropping = (3,0))(total_input)
#     target_input = Cropping0D(cropping = (0, 48))(total_input)    
    
#     x = muscle_input
#     for i in range(hidden_layers):
#         x = Dense(hidden_size)(x)
#         x = Activation('relu')(x)
#     x = Concatenate(axis=-1)([target_input, x])
    
    
#     for i in range(hidden_layers):
#         x = Dense(hidden_size)(x)
#         x = Activation('relu')(x)
#     x = Dense(shape_out)(x)
#     x = Activation('sigmoid')(x)
    
    
#     actor = Model(inputs=[total_input], outputs=x)
#     return actor

def q_nn(nb_obs, nb_act, hidden_layers = 3, hidden_size = 64):
    action_input = Input(shape=(nb_act, ), name='action_input')
    observation_input = Input(shape=(1,) + (nb_obs, ), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    for i in range(hidden_layers):
        x = Dense(hidden_size)(x)
        x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return critic, action_input

# Load arm environment
from osim.env import Arm3DVecEnv
env = Arm3DVecEnv(False)

obs = env.reset()
obs = env.step(env.action_space.sample())

path1 = 'C:/Users/wrest/Documents/NNWeights/weights'
# Create networks for DDPG
# Next, we build a very simple model.
actor = policy_nn(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers = 3, hidden_size = 31)

print(actor.summary())

plot_model(actor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

qfunc = q_nn(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers = 3, hidden_size = 64)
print(qfunc[0].summary())
plot_model(qfunc[0], to_file='model_plot_critic.png', show_shapes=True, show_layer_names=True)

opt = Adam(lr=.001, clipnorm=1.)
a1 =  obs[0]
b1 = np.array(a1)
c1 = np.reshape(b1, [1,51])
        
#action = actor.predict(c1)

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.action_space.shape)
agent = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actor, critic=qfunc[0], critic_action_input=qfunc[1],
                  memory=memory, nb_steps_warmup_critic=40, nb_steps_warmup_actor=40,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
agent.compile(opt, metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely by
# stopping the notebook
# agent.load_weights(path1)

agent.fit(env, nb_steps=100000, visualize=False, verbose=0, nb_max_episode_steps=200, log_interval=10000)
# After training is done, we save the final weights.
agent.save_weights(filepath = path1, overwrite=True)


agent.load_weights(path1)
# Finally, evaluate our algorithm for 2 episodes.
agent.test(env, nb_episodes=2, visualize=False, nb_max_episode_steps=1000)