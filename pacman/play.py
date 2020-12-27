# Load a saved agent policy for the crtpole-v0 environment, then run a
# few episodes with the trained policy and a random policy. create video files
# with the rendered environment
import imageio
from absl import logging
import os
import datetime

import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies import tf_policy
from tf_agents.trajectories.policy_step import PolicyStep


import config
import env_loader

# magic trick for "Could not create cuDNN handle when convnets are used"
# https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-613039907
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def create_video(py_environment: PyEnvironment, tf_environment: TFPyEnvironment,
        policy: tf_policy, num_episodes=10, video_filename='imageio.mp4'):
  print("Generating video %s" % video_filename)
  with imageio.get_writer(video_filename, fps=60) as video:
    for episode in range(num_episodes):
      episode_return = 0.0
      time_step = tf_environment.reset()
      video.append_data(py_environment.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)			
        time_step = tf_environment.step(action_step.action)
        episode_return += time_step.reward
        video.append_data(py_environment.render())
      print(f"Generated episode {episode} of {num_episodes}. Return:{episode_return} ")

now = datetime.datetime.now()

# Load the cartpole environment
eval_py_env = env_loader.load_env(mode='nonskipping')
eval_tf_env = TFPyEnvironment(eval_py_env)

# Load the saved policy
policy_dir = config.POLICIES_PATH + config.EVAL_POLICY_DIR
saved_policy = tf.saved_model.load(policy_dir)

create_video(
    eval_py_env,
    eval_tf_env, 
    saved_policy, 
    num_episodes=config.NUM_VIDEO_EPISODES, 
    video_filename=os.path.join(config.VIDEOS_PATH, "video_%s.mp4" % config.EVAL_POLICY_DIR))

