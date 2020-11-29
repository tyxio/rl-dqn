import gym
import time
import numpy as np
import torch
import gym.spaces

import config
import gym_wrappers as gw
import dqn

FPS = 25

# If in Colab: Tunning the image rendering in colab
#
# Taken from
# https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7
'''
!apt-get install -y xvfb x11-utils

!pip install pyvirtualdisplay==0.2.* \
             PyOpenGL==3.1.* \
             PyOpenGL-accelerate==3.1.*

!pip install gym[box2d]==0.17.*

import pyvirtualdisplay

_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()
'''

# If on Windows: install FFmpeg
# http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# https://www.gyan.dev/ffmpeg/builds/

# Taken (partially) from
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/03_dqn_play.py

model = 'PongNoFrameskip-v4-best.dat'
record_folder = "video"
visualize = True

env = gw.make_env(config.DEFAULT_ENV_NAME)
if record_folder:
    env = gym.wrappers.Monitor(env, record_folder, force=True)

net = dqn.DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(
    model, map_location=lambda storage, loc: storage))

state = env.reset()
total_reward = 0.0

while True:
    start_ts = time.time()
    if visualize:
        env.render()
    state_v = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)

    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
    if visualize:
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
print("Total reward: %.2f" % total_reward)

if record_folder:
    env.close()
