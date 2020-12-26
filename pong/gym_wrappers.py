
#
# OpenAI Gym Wrappers
#
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py

import cv2
import numpy as np
import collections
import gym
import gym.spaces

import config as cfg

test_env = gym.make(cfg.DEFAULT_ENV_NAME)
print("action_space:", test_env.action_space.n)
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
print("action_meanings:", test_env.unwrapped.get_action_meanings())
print("observation_space:", test_env.observation_space.shape)


# Pong require a user to press the FIRE button to start the game.
# The following code corresponds to the wrapper FireResetEnvthat presses
#  the FIRE button in environments that require that for the game to start
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

# Codes a couple of important transformations for Pong:
#
# On one hand, it allows us to speed up significantly the training by
# applying max to N observations (four by default) and returns this as
# an observation for the step. This is because on intermediate frames,
# the chosen action is simply repeated and we can make an action decision
# every N steps as processing every frame with a Neural Network is quite
# a demanding operation, but the difference between consequent frames is
# usually minor.
#
# On the other hand, it takes the maximum of every pixel in the last
# two frames and using it as an observation. Some Atari games have a flickering
# effect (when the game draws different portions of the screen on even and odd
# frames, a normal practice among Atari 2600 developers to increase the
# complexity of the game’s sprites), which is due to the platform’s limitation.
# For the human eye, such quick changes are not visible, but they can confuse
# a Neural Network.


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

# Before feeding the frames to the neural network every frame is scaled down
# from 210x160, with three color frames (RGB color channels), to a
# single-color 84 x84 image using a colorimetric grayscale conversion.
# Different approaches are possible. One of them is cropping non-relevant
# parts of the image and then scaling down as is done in the following code


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unkown resolution"

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

# Stacks several (usually four) subsequent frames together:


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype
        )

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low,
            dtype=self.dtype
        )
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

# The input shape of the tensor has a color channel as the last dimension,
# but PyTorch’s convolution layers assume the color channel to be the
# first dimension. This simple wrapper changes the shape of the observation
# from HWC (height, width, channel) to the CHW (channel, height, width)
# format required by PyTorch


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

# The screen obtained from the emulator is encoded as a tensor of bytes
# with values from 0 to 255, which is not the best representation for
# an NN. So, we need to convert the image into floats and rescale
# the values to the range [0.0…1.0]. This is done by the
# ScaledFloatFrame wrapper:


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

# wrap the wrappers:


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
