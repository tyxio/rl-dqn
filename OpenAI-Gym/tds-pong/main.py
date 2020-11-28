# https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
# https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_15_16_17_DQN_Pong.ipynb
#
# PacktPublishing Github:
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/tree/master/Chapter04
#
# Prerequisite Windows 10 :
# see https://github.com/openai/gym/issues/1726#issuecomment-550580367
# or https://stackoverflow.com/questions/63080326/could-not-find-module-atari-py-ale-interface-ale-c-dll-or-one-of-its-dependenc/64104353#64104353
#
# python -c "import atari_py;print(atari_py.list_games())"

# FloydHub:
# floyd login
# floyd init rl-dqn
# floyd run --gpu --env pytorch "python main.py"

import datetime
import numpy as np
import torch
import torch.nn as nn           # Pytorch neural network package
import torch.optim as optim     # Pytorch optimization package
from torch.utils.tensorboard import SummaryWriter

import config
import gym_wrappers as gw
import dqn
import experience_replay as xr
import agent as ag
import warnings

print(torch.__version__)

_device = 'cuda'
device = torch.device(_device)

warnings.filterwarnings("ignore", category=UserWarning)

# ***************************************
# Main training Loop
# ***************************************

gw.make_env(config.DEFAULT_ENV_NAME)env = 
writer = SummaryWriter(comment="-" + config.DEFAULT_ENV_NAME)

# the main DQN neural network that we are going to train
net = dqn.DQN(
    env.observation_space.shape,
    env.action_space.n
).to(device)
print(net)

target_net = dqn.DQN(
    env.observation_space.shape,
    env.action_space.n
).to(device)

# create the experience replay buffer of the required size and pass
#  it to the agent
buffer = xr.ExperienceReplay(config.replay_size)
agent = ag.Agent(env, buffer)

epsilon = config.eps_start

# create an optimizer, a buffer for full episode rewards, a counter of
# frames and a variable to track the best mean reward reached (because
# every time the mean reward beats the record, we will save the model
# in a file)
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
total_rewards = []
frame_idx = 0

best_mean_reward = None

print(">>>Training starts at ", datetime.datetime.now())
while True:  # while not converged
    frame_idx += 1

    #
    # Sample phase: Fill the replay buffer with agent experiences
    #

    # epsilon decay
    epsilon = max(epsilon * config.eps_decay, config.eps_min)

    # The Agent chooses an action and makes a single step in the Environment.
    # It observes the reward and the next state s' and stores the transition
    # (s, a, r, s', done) in the experience memory replay
    # (returns a non-None result only if this step is the final step in the
    # episode)
    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)

        # calculate and report the mean reward for the last 100 episodes
        mean_reward = np.mean(total_rewards[-100:])
        print("%d: %d games, mean reward %.3f, (epsilon %.2f)" %
              (frame_idx, len(total_rewards), mean_reward, epsilon))

        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)

        # After, every time our mean reward for the last 100 episodes reaches
        #  a maximum, we report this in the console and save the current model
        #  parameters in a file
        if best_mean_reward is None or mean_reward > best_mean_reward:
            torch.save(
                net.state_dict(),
                config.DEFAULT_ENV_NAME + "-best.dat"
            )
            best_mean_reward = mean_reward
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (best_mean_reward))

        # if this mean rewards exceed the specified MEAN_REWARD_BOUND
        # (19.0 in our case) then we stop training
        if mean_reward > config.MEAN_REWARD_BOUND:
            print("Sloved in %d frames!" % frame_idx)
            break

    if len(buffer) < config.replay_start_size:
        continue

    #
    # Learning phase
    #

    # sample a random mini-batch of transactions from the replay memory
    batch = buffer.sample(config.batch_size)
    states, actions, rewards, dones, next_states = batch

    # wraps individual numpy arrays with batch data in PyTorch tensors
    # and copies them to GPU. The code is written in a form to maximally
    # exploit the capabilities of the GPU by processing (in parallel)
    # all batch samples with vector operations
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_mask = torch.ByteTensor(dones).to(device)

    # pass observations to the first model and extract the specific Q-values
    # for the taken actions using the gather() tensor operation. The first
    # argument to this function call is a dimension index that we want to
    # perform gathering on. In this case, it is equal to 1, because it
    # corresponds to actions dimension:
    # bug in gather. See:
    if (_device == 'cpu'):
        # https://github.com/pytorch/pytorch/issues/37996#issuecomment-625363813
        t = torch.tensor(actions_v.unsqueeze(-1), dtype=torch.int64)
        state_action_values = net(states_v).gather(1, t).squeeze(-1)
    else:
        state_action_values = net(states_v). \
            gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Now that we have calculated the state-action values for every transition
    # in the replay buffer, we need to calculate target “y” for every
    # transition in the replay buffer too. Both vectors are the ones we
    # will use in the loss function. To do this, remember that we must
    # use the target network.
    next_state_values = target_net(next_states_v).max(1)[0]
    # Function max() returns both maximum values and indices of those values
    # (so it calculates both max and argmax). Because in this case, we are
    # interested only in values, we take the first entry of the result.

    # Remember that if the transition in the batch is from the last step in
    # the episode, then our value of the action doesn’t have a discounted
    # reward of the next state, as there is no next state from which to
    # gather the reward:
    next_state_values[dones_mask] = 0.0

    # Although we cannot go into detail, it is important to highlight that the
    # calculation of the next state value by the target neural network
    # shouldn’t affect gradients. To achieve this, we use thedetach() function
    # of the PyTorch tensor, which makes a copy of it without connection
    # to the parent’s operation, to prevent gradients from flowing into
    # the target network’s graph:
    next_state_values = next_state_values.detach()

    # Now, we can calculate the Bellman approximation value for the vector of
    # targets (“y”), that is the vector of the expected state-action value for
    # every transition in the replay buffer:
    expected_state_action_values = next_state_values * config.gamma + rewards_v

    # We have all the information required to calculate the mean squared
    # error loss:
    loss_t = nn.MSELoss()(state_action_values,
                          expected_state_action_values)

    # The next piece of the training loop updates the main neural network
    # using the SGD algorithm by minimizing the loss:
    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()

    # Finally, the last line of the code syncs parameters from our main
    # DQN network to the target DQN network every sync_target_frames:
    if frame_idx % config.sync_target_frames == 0:
        target_net.load_state_dict(net.state_dict())

writer.close()

print(">>>Training ends at ", datetime.datetime.now())

# tensorboard --logdir=runs
