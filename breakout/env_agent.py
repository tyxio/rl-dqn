import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent

import config

def create_qnetwork(env):
    '''
    Create a small class to normalize the observations. 
    Images are stored using bytes from 0 to 255 to use 
    less RAM, but we want to pass floats from 0.0 to 1.0 
    to the neural network:
    '''
    preprocessing_layer = keras.layers.Lambda(
                            lambda obs: tf.cast(obs, np.float32) / 255.)
    
    conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    
    fc_layer_params=[512]

    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)

    return q_net

def create_dqn_agent(env, q_net):
 
    # see TF-agents issue #113
    #optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
    #                                     epsilon=0.00001, centered=True)

    train_step = tf.Variable(0)
    update_period = config.UPDATE_PERIOD # run a training step every 4 collect steps
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
                                        epsilon=0.00001, centered=True)
    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
        end_learning_rate=0.01) # final ε
    
    agent = DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000, # <=> 32,000 ALE frames
        td_errors_loss_fn=keras.losses.Huber(reduction="none"),
        gamma=0.99, # discount factor
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step))
    
    return agent

def create_network_and_agent(env):

    qnet = create_qnetwork(env)
    agent = create_dqn_agent(env, qnet)
    agent.initialize()

    return agent
