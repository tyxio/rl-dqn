# rl-dqn
 Reinforcement Learning - DQN examples

## OpenAI Gym PONG

References:  
* https://torres.ai/deep-reinforcement-learning-explained-series/,  

* https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_15_16_17_DQN_Pong.ipynb



<img src="OpenAI-Gym\tds-pong\images\epsilon.png" alt="epsilon" width="250"/>
<img src="OpenAI-Gym\tds-pong\images\reward.png" alt="reward" width="250"/>

<img src="OpenAI-Gym\tds-pong\images\logs-end.png" alt="reward" width="500"/>

<img src="OpenAI-Gym\tds-pong\images\final_score.PNG" alt="final_score" width="200"/>

## Train a cartpole DQN with TF-Agents

This second example shows how to train a DQN (Deep Q Networks) agent on the cartpole environment using the TF-Agents library. It is the "Hello World" project for TF-Agents. I choose this examnple because it is simple to train, and so I could focus on the TF-Agents architecture.

References:

- https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
- https://rubikscode.net/2019/12/23/ultimate-guide-to-deep-q-learning-with-tf-agents/

Training this environment using a simple network with one hidden layer with 100 neurons yields the maximum average return of 200 after 16000 iterations:

<img src="cartpole/images/logs.PNG" alt="epsilon" width="400"/>

<img src="cartpole/images/avg-return.PNG" alt="epsilon" width="300"/>

The training took approximatly 25 min. on my computer (I7, 1 GPU). 

A video showing the trained agent over 5 episodes was created (available in the folder videos):

[![Alt Text](https://img.youtube.com/vi/azqA_WNW0-k/0.jpg)](https://youtu.be/azqA_WNW0-k)

A video was also generated using a random policy: 

[![Alt Text](https://img.youtube.com/vi/0sNOnWcRHeI/0.jpg)](https://youtu.be/0sNOnWcRHeI))

The code for defining the TF-Agent and training it is in main.py. The code for generating these videos is in play.py 

Conclusions:

1. A DQN can be trained successfully for the cartpole environment
2. Using a RL framework like TF-Agents simplifies greatly the code. Of course, there is a learning curve...

## Solving Atari Breakout with TF-Agents DQN

This example uses TF-Agents to train an agent to play Breakout, the famous Atari game, using the DQN algorithm. I used the OpenAI Gym Breakout-v0 environment (https://gym.openai.com/envs/Breakout-v0/). The code is based on the cartpole example above. But this time, I used a CNN Network to train the agent as the observations of the environment are screenshots of the Atari screen. 

The Atari Breakout environment and its DQN training with TF-Agents are described in great details in Aurélien Géron's Book: "Hands-On Machine Learning with Scikit-Learn, Keras and tensorflow, 2nd Edition". Its github repository is https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb

Training an Atari breakout agent requires much more ressources thant the cartpole environment. A. Géron suggests 10E7 steps. On my machine, it would require 8 days... So I have limited the training to 10E6 steps. After 16 hours of calculation, the average return approaches 30 which is still far from 200 that can be obtained with more ressources:

<img src="breakout/images/averageReturnMetric.PNG" alt="epsilon" width="400"/>

But I consider it is not bad because it validates the algorithm and we can see already a big improvement over a random policy:

[![Alt Text](https://img.youtube.com/vi/7_hV9potEDg/0.jpg)](https://youtu.be/7_hV9potEDg))


