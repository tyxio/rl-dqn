# Load a saved agent policy for the crtpole-v0 environment, then run a
# few episodes with the trained policy and a random policy. create video files
# with the rendered environment
import base64
import imageio
import IPython
import os
import datetime

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

print(tf.__version__)

# a function to embed videos in the notebook.


def embed_mp4(filename):
    """Embeds an mp4 file."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

# Run a few episodes with the specified policy and stored the frames
# in a video file


def run_episodes_and_create_video(policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    return embed_mp4(filename)


now = datetime.datetime.now()

# Load the cartpole environment
eval_py_env = suite_gym.load('CartPole-v0')
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Load the saved policy
policy_dir = 'policies/cartpole'
saved_policy = tf.saved_model.load(policy_dir)

# Prepare the videos folder
video_dir = 'videos/cartpole'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
video_file = video_dir + "/trained-agent-" + now.strftime("%m%d%Y-%H%M%S")

# Iterate through a few episodes of the Cartpole game with the agent.
# The underlying Python environment (the one "inside" the TensorFlow
# environment wrapper) provides a render() method, which outputs an
# image of the environment state. These can be collected into a video.
run_episodes_and_create_video(saved_policy, video_file)

# For fun compare the trained agent (above) to an agent moving randomly.
# (It does not do as well.)

# Policies can be created independently of agents. For example, use
# tf_agents.policies.random_tf_policy to create a policy which will
# randomly select an action for each time_step.
random_policy = random_tf_policy.RandomTFPolicy(eval_env.time_step_spec(),
                                                eval_env.action_spec())

video_file = video_dir + "/random-agent-" + now.strftime("%m%d%Y-%H%M%S")
run_episodes_and_create_video(random_policy, video_file)
