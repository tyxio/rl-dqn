import config
import utils
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4


def load_env(mode='default'):
    env = None
    if (mode == 'default'):
        env = suite_gym.load(config.ENV_NAME)
    elif (mode == 'nonskipping'):
        max_episode_steps = 27000  # <=> 108k ALE frames since 1 step = 4 frames
        env = suite_atari.load(
            config.ENV_NONSKIPPING_NAME,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=[AtariPreprocessing, FrameStack4]
        )
    return env


def check_env(env):
    env.seed(42)

    print('Reset env:')
    ts0 = env.reset()
    print(ts0)

    print('Fire 1 step:')
    ts1 = env.step(np.array(1))
    print(ts1)

    # The observations are screenshots of the Atari screen,
    # represented as NumPy arrays of shape [210, 210, 3]
    print('observation spec:')
    print(env.observation_spec())

    print('action spec:')
    print(env.action_spec())

    print('time step spec:')
    print(env.time_step_spec())

    print('current time step:')
    print(env.current_time_step())

    print('possible actions:')
    print(env.gym.get_action_meanings())


def play_steps(env, nsteps=4):
    env.seed(42)
    env.reset()
    time_step = env.step(np.array(1))  # FIRE
    for _ in range(nsteps):
        time_step = env.step(np.array(3))  # LEFT
    return time_step

def render(env):
    # to render an environment, you can use the mode 'human'
    #img = env.render(mode='human')
    # to get back the image in the form of a NumPy array, use mode='rgb_array'
    img = env.render(mode='rgb_array')

    plt.figure(figsize=(6, 8))
    plt.imshow(img)
    plt.axis('off')
    utils.save_fig(config.ENV_SHORT_NAME + '_step1')
    plt.show()

def render_observation(time_step):
    plt.figure(figsize=(4,4))
    utils.plot_observation(time_step.observation)
    utils.save_fig("preprocessed_" + config.ENV_SHORT_NAME + "_plot")
    plt.show()
