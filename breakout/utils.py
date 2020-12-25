import config
import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

now = datetime.datetime.now()

os.makedirs(config.IMAGES_PATH, exist_ok=True)

#
# Plotting
#
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(config.IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")

def plot_trajectories(trajectories):
    plt.figure(figsize=(10, 6.8))
    for row in range(2):
        for col in range(3):
            plt.subplot(2, 3, row * 3 + col + 1)
            plot_observation(trajectories.observation[row, col].numpy())
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0.02)
    save_fig(config.ENV_SHORT_NAME + "sub_episodes_plot")
    plt.show()  

#
# Tensorboard
#
writer = tf.summary.create_file_writer(
    config.PROJECT_ROOT_DIR + "/runs/" + now.strftime("%m%d%Y-%H%M%S")
    )

def write_summary(name, value, step):
    with writer.as_default():
        tf.summary.scalar(name=name, data=value, step=step)

#
# System
#
def check_gpu():
    if not tf.config.list_physical_devices('GPU'):
        print("No GPU was detected. CNNs can be very slow without a GPU.")
    else:
        print("Lucky guy, you have a GPU!")   
        print(tf.__version__)
        # Fix GPU memory issue after the NVDIA driver was updaded on my machine
        # https://github.com/tensorflow/tensorflow/issues/25403#issuecomment-708787479
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            assert tf.config.experimental.get_memory_growth(physical_devices[0])
        except:
            pass

def print_time_stats(t0, iteration):
    if (iteration == 0):
        return
    t1 = (datetime.datetime.now() - t0).total_seconds()
    secs_per_iteration = t1 / iteration  
    remaing_secs = (config.TRAINING_STEPS - iteration) * secs_per_iteration
    remaining_hours = remaing_secs /3600
    print(f" 1000 iter: {secs_per_iteration*1000} s - Remaining:{remaining_hours} h")
