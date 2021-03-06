# My Journey into Reinforcement Learning and Deep Q-Learning

This repository contains projects that I have choosen to learn __Reinforcement Learning__ (RL) with __Deep Q-Network__ (DQN). I had 2 main goals:

1. Learn how DQN can be implemented in Python
2. Learn a RL framework that makes this implementation easier. I have choosen __TF-Agents__ as I have worked mainly with Tensorflow up to now.

You will find 4 projects:

1. The __PONG__ game, a first project to understand RL and DQN without a framework;
2. The __Cartpole__ environment, a simple RL problem I have selected to approach TF-Agents;
3. The __Atari Breakout__ game, a more complex problem to understand the advantages of using a RL framwork like TF-Agents;
4. The __Pacman__ game, that allowed me to reuse 98% of the code developed for training Breakout

## First DQN: the PONG game

For my first RL DQN project, I have decided to implement the Deep Q-Learning algorithm, without using a RL framework, to train an agent to play the PONG game.

<img src="pong/images/final_score.png" alt="final_score" width="150"/>
 
 My idea was to understand how DQN can be implemented, particularly the replay buffer and the target model. I read different tutorials and found that Jordi Tores' series "Deep Reinforcement Learning Explained" was the best to get a solid training on RL DQN. 


References:  
* https://torres.ai/deep-reinforcement-learning-explained-series

* https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_15_16_17_DQN_Pong.ipynb

I trained the agent network with 3 convolutional layers over about 1M steps to reach the maximun reward of 20:

<img src="pong/images/reward.png" alt="reward" width="250"/>

<img src="pong/images/logs-end.png" alt="reward" width="500"/>
 

Here is a video showing that the DQN agent was well trained (he is on the right side):

<a href="https://youtu.be/EFc5bdf8fos" target="_blank"><img src="https://img.youtube.com/vi/EFc5bdf8fos/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="5" /></a>

## Train a Cartpole DQN with TF-Agents

The second example shows how to train a DQN (Deep Q Networks) agent on the cartpole environment using the TF-Agents library. It is the "Hello World" project for TF-Agents. I choose this example because it is simple to train, and so I could focus on the TF-Agents architecture.

References:

- https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
- https://rubikscode.net/2019/12/23/ultimate-guide-to-deep-q-learning-with-tf-agents/

Training this environment using a simple network with one hidden layer with 100 neurons yields the maximum average return of 200 after 16000 iterations:

<img src="cartpole/images/logs.PNG" alt="epsilon" width="400"/>

<img src="cartpole/images/avg-return.PNG" alt="epsilon" width="300"/>

The training took approximatly 25 min. on my computer (I7, 1 GPU). 

A video showing the trained agent over 5 episodes was created. You can compare with an agent trained with a random policy (the videos open in YouTube):


| trained cartpole (5 episodes)  |  random policy (5 episodes) |
|---|---|
|  <a href="https://youtu.be/azqA_WNW0-k" target="_blank"><img src="https://img.youtube.com/vi/azqA_WNW0-k/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="5" /></a>  | <a href="https://youtu.be/0sNOnWcRHeI" target="_blank"><img src="https://img.youtube.com/vi/0sNOnWcRHeI/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="5" /></a> 


Conclusions:

1. A DQN can be trained successfully for the cartpole environment
2. Using a RL framework like TF-Agents simplifies greatly the code. Of course, there is a learning curve...

## Solving Atari Breakout with TF-Agents DQN

This example uses TF-Agents to train an agent to play Breakout, the famous Atari game, using the DQN algorithm. I used the OpenAI Gym Breakout-v0 environment (https://gym.openai.com/envs/Breakout-v0/). The code is based on the cartpole example above. But this time, I used a CNN Network to train the agent as the observations of the environment are screenshots of the Atari screen. 

The Atari Breakout environment and its DQN training with TF-Agents are described in great details in Aurélien Géron's Book: "Hands-On Machine Learning with Scikit-Learn, Keras and tensorflow, 2nd Edition". Its github repository is https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb

Training an Atari breakout agent requires much more ressources thant the cartpole environment. A. Géron suggests 10E7 steps. On my machine, it would require 8 days... So I have limited the training to 10E6 steps. After 16 hours of calculation, the average return approaches 30 which is still far from 200 that can be obtained with more ressources:

<img src="breakout/images/averageReturnMetric.png" alt="epsilon" width="300"/>

But I consider it is not bad because it validates the algorithm and we can see already a big improvement over a random policy:


<a href="https://youtu.be/7_hV9potEDg" target="_blank"><img src="https://img.youtube.com/vi/7_hV9potEDg/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="5" /></a>

## Solving Pacman with TF-Agents DQN

Finally: Pacman. Well, it was much easier than I though thanks to the Breakout project I have done before. I have reused the code and changed the name of the OpenAI Gym environment to MsPacman-v0 (actually to MsPacmanNoFrameskip-v0 because the default Atari environment applies random frame skipping and max pooling and we must train on the raw, nonskipping variant).

Again, I was limited by the resources and trained Pacman over 1.8M steps on Google Cloud (see Appendix 2). Note that A. Geron recommends 10M  steps. An average return per episode of ~ 2300 is obtained. 

| Average Episode Length  |  Average Return |
|---|---|
|  <img src="pacman/images/averageEpisodeLengthMetric.png" alt="epsilon" width="300"/>  | <img src="pacman/images/averageReturnMetric.png" alt="epsilon" width="300"/>  

It's time for the video. Not perfect, but already better than what I could perform as a human agent:

 <a href="https://youtu.be/r_ykSNO9dbc" target="_blank"><img src="https://img.youtube.com/vi/r_ykSNO9dbc/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="5" /></a>

 ## The Next Journey

The 4 examples I have described above gave me a good understanding of how RL DQN can be implemented using the TF-Agents framework. There is of course still a lot to discover as the model is only part of what it takes to succeed with a machine learning project. As a software enginneer, I'd like to experiment more how the scripts that I have run on my personal latptop or on Google Cloud can be executed in an environment with multiple GPUs. Indeed, running training models during days is not viable outside the personal experimention or academic domain. In production, parallel processing distributed among multiple CPUs/GPUs are necessary to reduce the time from lab to market. For that, I believe that [Dask](https://dask.org/) and [RAPIDS](https://rapids.ai/) are the next things to learn. 

### Appendix 1 - Run a tf-agents script on Ubuntu 18

1. See https://www.tensorflow.org/install/pip#ubuntu-macos
2. pip3 install --upgrade numpy tf-agents
4. sudo apt-get install python3-matplotlib python-opencv
5. pip3 install gym>=0.17.3 atari-py 

### Appendix 2 - Run a tf-agents script on Google Cloud

1. Choose a machine with GPU and enough memory e.g. n1-highmem-8 (8 vCPUs, 52 GB memory) with 1 x NVIDIA Tesla P4
2. SSH
3. conda activate base
4. cd cd rl-dqn/
5. python3 ./breakout/main.py

A PDF of this readme page can be generated using [grip](https://github.com/joeyespo/grip)