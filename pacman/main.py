'''
Using TF-Agents to Beat Breakout
Let's use TF-Agents to create an agent that will learn to play Breakout

References:
    - Book of A. Geron: Hands-On Machine Learning with Scikit-Learn, Keras and Tensorflow (2nd edition)
    - A. Geron' gitub https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb

Run this file as a FloydHub job in project rl-dqn (already created):
    pip install -U floyd-cli
    floyd login
    floyd init rl-dqn
    floyd run --gpu --env tensorflow-2.2 'python main.py'
When the run is complete, download the results (job 5):
floyd data clone philhu/projects/rl-dqn/5
'''

#from breakout.utils import plot_trajectories
import numpy as np
import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# tf-agents imports
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.policies import policy_saver
from tf_agents.utils.common import function
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics

import logging

# local imports
import config
import utils
import env_loader
import env_agent

# magic trick for "Could not create cuDNN handle when convnets are used"
# https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-613039907
#utils.check_gpu()
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

tf.random.set_seed(42)
np.random.seed(42)

print(">>>Training starts at ", utils.now)

# Load the default gym environment (just to see it)
env = env_loader.load_env()
env_loader.check_env(env) # optional
#env_loader.render(env)    # to comment on remote machine

# specify the environment wrappers
#env_wrappers.list_tfagent_wrappers() # optional

# Create the Atari environment with stacks of 4 frames
# We will use this environment for training
env = env_loader.load_env(mode='nonskipping')

# Play a few steps just to see what happens:
#time_step = env_loader.play_steps(env)
#env_loader.render_observation(time_step)

#  Convert the Python environment to a TF environment:
tf_env = TFPyEnvironment(env)

# Create the Q-Network and the DQN Agent
agent = env_agent.create_network_and_agent(tf_env)

# Create the replay buffer (this may use a lot of RAM, so please 
# reduce the buffer size if you get an out-of-memory error):

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=config.REPLAY_BUFFER_MAX_LENGTH)

replay_buffer_observer = replay_buffer.add_batch

'''
Create a simple custom observer that counts and displays the 
number of times it is called (except when it is passed a 
trajectory that represents the boundary between two episodes, 
as this does not count as a step):
'''
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

'''
Training Metrics
https://www.tensorflow.org/agents/api_docs/python/tf_agents/metrics/tf_metrics
'''
train_metrics = [
    # Counts the number of episodes in the environment.
    tf_metrics.NumberOfEpisodes(),
    # Counts the number of steps taken in the environment.
    tf_metrics.EnvironmentSteps(),
    # Metric to compute the average return. (see comment on p 656)
    tf_metrics.AverageReturnMetric(),
    # Metric to compute the average episode length.
    tf_metrics.AverageEpisodeLengthMetric(),
]
# At any time you can get the value of each of these metrics by calling its result() method.
# (e.g. train_metrics[0].result()). Alternatively, you can log all metrics by calling 
# log_metrics:
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

'''
Collect Driver
https://www.tensorflow.org/agents/tutorials/4_drivers_tutorial

A common pattern in reinforcement learning is to execute a policy in an environment 
for a specified number of steps or episodes. This happens, for example, during 
data collection, evaluation and generating a video of the agent.

While this is relatively straightforward to write in python, it is much more 
complex to write and debug in TensorFlow because it involves tf.while loops, 
tf.cond and tf.control_dependencies. Therefore we abstract this notion of 
a run loop into a class called driver, and provide well tested implementations 
both in Python and TensorFlow.

Additionally, the data encountered by the driver at each step is saved 
in a named tuple called Trajectory and broadcast to a set of observers 
such as replay buffers and metrics. This data includes the observation 
from the environment, the action recommended by the policy, the reward 
obtained, the type of the current and the next step, etc.

We currently have 2 TensorFlow drivers: DynamicStepDriver, which terminates after a 
given number of (valid) environment steps and DynamicEpisodeDriver, which 
terminates after a given number of episodes. We want to collect experience for
4 steps for each training iteration (as was done in the 2015 DQN paper)
'''

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=config.UPDATE_PERIOD) # collect 4 steps for each training iteration

'''
We could now run the collect_driver by calling its run() method, ut it is best
to warm up the replay buffer with experiences collected using a purely random policy.
For this, we can use the RandomTFPolicy class and create a second driver that will run
the policy for 20000 steps (which is equivalent to 80000 simulator frames, as was done in 
the 2015 DQN paper). We use ShowProgress to display the progress:
'''
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(config.RANDOM_POLICY_PREFILL_STEPS)],
    num_steps=config.RANDOM_POLICY_PREFILL_STEPS) # <=> 80,000 ALE frames

print("warm up the replay buffer with experiences collected using a random policy...")
final_time_step, final_policy_state = init_driver.run()

# Let's sample 2 sub-episodes, with 3 time steps each and display them: (p 658)
tf.random.set_seed(888) # chosen to show an example of trajectory at the end of an episode

trajectories, buffer_info = replay_buffer.get_next(
    sample_batch_size=2, num_steps=3)

print(trajectories._fields)
print(trajectories.observation.shape)
time_steps, action_steps, next_time_steps = to_transition(trajectories)
print(time_steps.observation.shape)
print(trajectories.step_type.numpy())

# comment the line below during training on a remote host
#utils.plot_trajectories(trajectories)

'''
For the main training loop, instead of calling the get_next() method, we will use
a tf.data.Dataset. This way, we can benefit from the power of the data API (parallelism
and prefetching). For this, we call the replay buffer's as_dataset() method.

We will sample batches of 64 trajectories at each training step, each with 2 steps
(i.e. 2 steps = 1 full transition, including the next step's observation). This
dataset will process 3 elements in parallel, and prefetch 3 batches.

'''
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)


# To speed up training, convert the main functions to TF Functions. 
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# we will save the agent policy periodically
def save_agent_policy():
    now = datetime.datetime.now()
    policy_dir = config.POLICIES_PATH + now.strftime("%m%d%Y-%H%M%S")
    os.mkdir(policy_dir) 
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)
    print(">>>Policy saved in ", policy_dir)

'''
And now we are ready to run the main loop!
'''

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % config.TRAINING_LOG_INTERVAL == 0:
            print("\r")
            log_metrics(train_metrics)
        if iteration % config.TRAINING_SAVE_POLICY_INTERVAL == 0:
            save_agent_policy()
        if iteration % config.TRAINING_LOG_MEASURES_INTERVAL == 0:
            # calculate and report the total return over 1 episode
            utils.write_summary("AverageReturnMetric", train_metrics[2].result(), iteration)
            utils.write_summary("AverageEpisodeLengthMetric", train_metrics[3].result(), iteration)
            utils.writer.flush()

    save_agent_policy()
    utils.writer.flush()

train_agent(n_iterations=config.TRAINING_STEPS)

print(">>>Training ends at ", datetime.datetime.now())
