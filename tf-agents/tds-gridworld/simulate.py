import matplotlib.pyplot as plt
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver

from gridworld import GridWorldEnv

# First we will load the Gridworld environments into a TimeLimit Wrapper
#  which terminates the game if 100 steps are reached.
train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)

# The results are then wrapped in the TF environment handler
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Then we will create the network, the optimizer and the Deep Q-Network (DQN) agent.
# Here we decide to create a network with a single hidden layer of 100 hidden nodes. 
# For this kind of simple problem, this setup should be more than sufficient. 
# I am using the Adam optimizer because in general this is the state of the art 
# over something like vanilla Stochastic Gradient Descent (SGD). Though for this 
# example I imagine SGD would work as well. Feel free to test this out and let me
#  know in the comments.
fc_layer_params = (100,)
learning_rate = 1e-5
replay_buffer_capacity = 100000
batch_size = 128

num_iterations = 10000
log_interval = 200
num_eval_episodes = 2 
eval_interval = 1000

# https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/q_network/QNetwork
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params = fc_layer_params
)

# https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/DqnAgent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

tf_agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network = q_net,
    optimizer = optimizer,
    td_errors_loss_fn = common.element_wise_squared_loss,
    train_step_counter = train_step_counter
)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

# Next we create the replay buffer and replay observer. The replay buffer is
# used to contain the observation and action pairs so they can be used 
# for training
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = tf_agent.collect_data_spec, # the shape of a row that is contained in the buffer
    batch_size = train_env.batch_size,      # the size of the batch that is saved in the replay buffer
    max_length = replay_buffer_capacity     # how many records the buffer can hold before starting 
                                            # overwrite the oldest ones with the newest entries
)

print("Batch Size: {}".format(train_env.batch_size))

replay_observer = [replay_buffer.add_batch]

# Then we create a dataset out of the replay buffer that we can 
# iterate through and train the agent.
dataset = replay_buffer.as_dataset(
    num_parallel_calls = 3,         # the number of items to process in paralell
    sample_batch_size = batch_size,  # the number of items to pass to the Neural Network for training
    num_steps = 2                   # the number of consecutive items to return as sub batches in the returned batch
).prefetch(3)

iterator = iter(dataset)

# Then we want to create the driver which will simulate the agent in the game 
# and store the state, action, reward pairs in the replay buffer, 
# and keep track of some metrics.
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric()
]

driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers = replay_observer + train_metrics,
    num_steps = 1
)

# Then finally we come to the cycle of training where the driver 
# is run and the experience is drawn from the dataset and used 
# to train the agent

final_time_step, policy_state = driver.run()

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

episode_len = []
step_len = []
for i in range(num_iterations):
    final_time_step, _ = driver.run(final_time_step, policy_state)

    experience, _ = next(iterator)
    train_loss = tf_agent.train(experience=experience)
    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        episode_len.append(train_metrics[3].result().numpy())
        step_len.append(step)
        print('Average episode length: {0}'.format(train_metrics[3].result().numpy()))
    
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))

plt.plot(step_len, episode_len)
plt.xlabel('Episodes')
plt.ylabel('Average Episode Length (Steps)')
plt.show()