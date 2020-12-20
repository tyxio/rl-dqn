# https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
# https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial
# https://rubikscode.net/2019/12/23/ultimate-guide-to-deep-q-learning-with-tf-agents/
import datetime
import os
import tensorflow as tf

'''
One of the classes we imported is DqnAgent, specific agent that can perform Deep Q-Learning. 
This is really cool and saves us a lot of time. Also we imported QNetwork class. 
This class is an abstraction of neural network that we use for learning. As you can see, 
as with transfer learning, this saves us a bunch of time. We also import suite_gym and 
tf_py_environment. The first module grants us access to training environments. Since 
all of these environments are implemented in Python, we need to wrap them up into TensorFlow. 
That is what tf_py_environment is used for. For experience replay, we use class 
TFUniformReplayBuffer and in this buffer we store trajectories. Trajectory is a tuple 
that contains state of the environment in some time step, action that agent should 
take it in that state and state in which the environment will be after defined action 
is performed. 
'''

from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks.q_network import QNetwork

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

print(tf.__version__)

# https://github.com/tensorflow/tensorflow/issues/25403#issuecomment-708787479
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
  pass

# Globals
MAX_RETURN = 30
COLLECTION_STEPS = 1
BATCH_SIZE = 64
EVAL_EPISODES = 10
EVAL_INTERVAL = 1000
LOG_INTERVAL = 100
learning_rate = 1e-3

now = datetime.datetime.now()
print(">>>Training starts at ", now)

# tensorboard writer
writer = tf.summary.create_file_writer(
    "cartpole/runs/" + now.strftime("%m%d%Y-%H%M%S")
    )


def write_summary(name, value, step):
    with writer.as_default():
        tf.summary.scalar(name=name, data=value, step=step)


# Usually two environments are instantiated: one for training
# and one for evaluation.
train_py_env = suite_gym.load('CartPole-v0')
evaluation_py_env = suite_gym.load('CartPole-v0')

'''
In the Cartpole environment:
= observation is an array of 4 floats:
 - the position and velocity of the cart
 - the angular position and velocity of the pole
= reward is a scalar float value
= action is a scalar integer with only two possible values:
- 0 — "move left"
- 1 — "move right"

Observation Spec:
BoundedArraySpec(shape=(4,), dtype=dtype('float32'), name='observation', 
 minimum=[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], 
 maximum=[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38])
Reward Spec:
ArraySpec(shape=(), dtype=dtype('float32'), name='reward')
Action Spec:
BoundedArraySpec(shape=(), dtype=dtype('int64'), name='action', minimum=0, maximum=1)
'''
print("Observation Spec:") 
print(train_py_env.time_step_spec().observation)
print("Reward Spec:")
print(train_py_env.time_step_spec().reward)
print("Action Spec:")
print(train_py_env.action_spec())


# The Cartpole environment, like most environments, is written in pure Python.
# This is converted to TensorFlow using the TFPyEnvironment wrapper.

# The original environment's API uses Numpy arrays. The TFPyEnvironment
# converts these to Tensors to make it compatible with Tensorflow agents
# and policies.
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
evaluation_env = tf_py_environment.TFPyEnvironment(evaluation_py_env)

# *******************
# Build the DQN agent
# *******************

'''
At the heart of a DQN Agent is a QNetwork, a neural network model that can learn to 
predict QValues (expected returns) for all actions, given an observation from the environment.
Use tf_agents.networks.q_network to create a QNetwork, passing in the observation_spec, 
action_spec, and a tuple describing the number and size of the model's hidden layers.
'''
hidden_layers = (100,)
q_network = QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=hidden_layers
)

'''
Now use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. In addition 
to the time_step_spec, action_spec and the QNetwork, the agent constructor 
also requires an optimizer (in this case, AdamOptimizer), a loss function, 
and an integer step counter.
'''
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_network,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

# *******************
# Metrics and Evaluation
# *******************

'''
The most common metric used to evaluate a policy is the average return. 
The return is the sum of rewards obtained while running a policy in an 
environment for an episode. Several episodes are run, creating an average return.
The following function computes the average return of a policy, given the policy, 
environment, and a number of episodes.
'''
def get_average_return(environment, policy, episodes=10):
    total_return = 0.0

    for _ in range(episodes):
        episode_return = 0.0
        time_step = environment.reset()

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step)
            episode_return += time_step.reward

        total_return += episode_return
    avg_return = total_return / episodes

    return avg_return.numpy()[0]

# ******************
# Experience replay
# ******************


class ExperienceReplay(object):
    # In the constructor of this class, we initialize replay buffer, 
    # which is an object of the class TFUniformReplayBuffer. If your agent 
    # is not getting good results, you can play with batch size and 
    # length of the buffer.
    def __init__(self, agent, environment):
        self._replay_buffer = TFUniformReplayBuffer(
            # For most agents, collect_data_spec is a named tuple called Trajectory, 
            # containing the specs for observations, actions, rewards, and other items.
            data_spec=agent.collect_data_spec,
            batch_size=environment.batch_size,
            max_length=50000
        )

        # we create an instance of RandomTFPolicy. This one is used to
        # fill the buffer with initial values, which is done by calling
        # the internal function _fill_buffer
        self._random_policy = RandomTFPolicy(
            train_env.time_step_spec(),
            environment.action_spec()
        )

        # fill the buffer with random initial values
        self._fill_buffer(train_env, self._random_policy, steps=100)

        '''
        The agent needs access to the replay buffer. This is provided by 
        creating an iterable tf.data.Dataset pipeline which will feed data to the agent.
        Each row of the replay buffer only stores a single observation step. But 
        since the DQN Agent needs both the current and next observation to compute 
        the loss, the dataset pipeline will sample two adjacent rows for each 
        item in the batch (num_steps=2).
        This dataset is also optimized by running parallel calls and prefetching data.
        '''
        self.dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=BATCH_SIZE,
            num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)

    def _fill_buffer(self, environment, policy, steps):
        for _ in range(steps):
            self.timestamp_data(environment, policy)

    # form a trajectory from the current state and the action defined by
    # the policy. This trajectory is stored in the buffer
    def timestamp_data(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        timestamp_trajectory = trajectory.from_transition(
            time_step, action_step, next_time_step)

        self._replay_buffer.add_batch(timestamp_trajectory)


experience_replay = ExperienceReplay(agent, train_env)

# ************************
# Training and Evaluation
# ************************

'''
Two things must happen during the training loop:
 - collect data from the environment
 - use that data to train the agent's neural network(s)
This example also periodicially evaluates the policy and prints the current score.
The following will take ~25 minutes to run.
'''

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_average_return(evaluation_env, agent.policy, EVAL_EPISODES)
write_summary("return", avg_return, 0)
returns = [avg_return]

n = 0
while True:  # while not converged
    n += 1

    # Collect a few steps using collect_policy and save to the
    # replay buffer.
    for _ in range(COLLECTION_STEPS):
        experience_replay.timestamp_data(train_env, agent.collect_policy)

    # sample a batch of data from the replay buffer and 
    # update the agent's network.
    experience, info = next(experience_replay.iterator)
    train_loss = agent.train(experience).loss

    if agent.train_step_counter.numpy() % LOG_INTERVAL == 0:
        # calculate and report the total return over 1 episode
        _return = get_average_return(
            evaluation_env, agent.policy, 1)
        write_summary("return", _return, n)
        write_summary("loss", tf.cast(train_loss, tf.int64), n)
        writer.flush()

    if agent.train_step_counter.numpy() % EVAL_INTERVAL == 0:
        # calculate and report the average total return over
        # EVAL_EPISODES episodes
        avg_return = get_average_return(
            evaluation_env, agent.policy, EVAL_EPISODES)
        returns.append(avg_return)

        write_summary("avg-return", avg_return, n)
        write_summary("loss", tf.cast(train_loss, tf.int64), n)
        writer.flush()

        print('Iteration {0} – Average Return = {1}, Loss = {2}.'.format(
            agent.train_step_counter.numpy(), avg_return, train_loss))

        # stop training if we reach the max return (200)
        if (avg_return >= MAX_RETURN):
            print("Solved in %d iterations!" % n)
            break

writer.flush()

# save policy

# prepare for saving the agent policy
policy_dir = 'cartpole/policies/'+ now.strftime("%m%d%Y-%H%M%S")
os.mkdir(policy_dir) 
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)

print(">>>Training ends at ", datetime.datetime.now())
print(">>>Policy saved in ", policy_dir)