import numpy as np
import torch

import experience_replay

# The agent performs these 3 actions:
#
# 1. Choose an action a from state s using policy eps-greedy(Q)
# 2. Agent takes action a, observe reward r, and next state s'
# 3. Store transition (s, a, r, s') in the experience replay memory D


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self.total_reward = 0.0
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='gpu'):
        done_reward = None

        # select action
        if np.random.random() < epsilon:
            # take the random action
            action = self.env.action_space.sample()
        else:
            # use the past model to obtain the Q-values for all possible
            # actions and choose the best
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # performs the step in the Environment to get the next observation
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # stores the observation in the experience replay buffer
        exp = experience_replay.Experience(
            self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        # handle the end-of-episode situation
        if is_done:
            done_reward = self.total_reward
            self._reset()

        # The result of the function is the total accumulated reward if we have
        # reached the end of the episode with this step, or None if not.
        return done_reward
