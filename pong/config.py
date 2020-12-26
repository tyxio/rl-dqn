DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'

# We will consider that the game has converged when our agent reaches
# an average of 19 games won (out of 21) in the last 100 games
MEAN_REWARD_BOUND = 19.0

gamma = 0.99         # the discount factor
batch_size = 32      # the minibatch size
# the maximum number of experiences stored in replay memory
replay_size = 10000

learning_rate = 1e-4

# indicates how frequently we sync model weights from the
# main DQN network to the target DQN network (how many
# frames in between syncing)
# the count of frames (experiences) to add to replay buffer
sync_target_frames = 1000

replay_start_size = 10000

# before starting training.
eps_start = 1.0
eps_decay = .999985
eps_min = 0.02
