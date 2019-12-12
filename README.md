# Project 1: Navigation

The very first project in Udacity DRL nanodegree program. The agent is suppose to pick all the yellow banana without interfering with the blue ones.

## Details of the unity agent :
		
__Unity brain name: BananaBrain__
* Number of Visual Observations (per agent): 0
* Vector Observation space type: continuous
* Vector Observation space size (per agent): 37
* Number of stacked Vector Observation: 1
* Vector Action space type: discrete
* Vector Action space size (per agent): 4

## Dependencies:

Python Environment can be set up based on the following instructions:

1. Create (and activate) a new environment with Python 3.6.
	* conda create --name drlnd python=3.6
	* source activate drlnd
	
## Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here][https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip]
Mac OSX: [click here][]https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
Windows (32-bit): [click here][https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip]
Windows (64-bit): [click here][https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip]

Place the file in the working folder and unzip the file

Start working with Navigation.ipynb

## Instructions:

Once the environment is set, we can run the Navigation.ipynb file. model.py contains neural network class used as a Q function and file dqn_agent.py contains agent code.
The python notebook can guide us to the results.
       
## Implement Learning Algorithm
The primary objective of the learning algorithm is to find an optimal policy—i.e., a policy that maximizes the reward for the agent. Since the effects of possible actions aren't known in advance, the optimal policy must be discovered by interacting with the environment and recording observations. Therefore, the agent "learns" the policy through a process of trial-and-error that iteratively maps various environment states to the actions that yield the highest reward. This type of algorithm is called Q-Learning.

The general approach is to implement a handful of different components, then run a series of tests to determine which combination of components and which hyperparameters yield the best results.

In the following sections, we'll describe each component of the algorithm in detail.

## Q-Function
The Q-function calculates the expected reward R for all possible actions A in all possible states S.

We can then define our optimal policy π* as the action that maximizes the Q-function for a given state across all possible states. The optimal Q-function Q*(s,a) maximizes the total expected reward for an agent starting in state s and choosing action a, then following the optimal policy for each subsequent state.

## Epsilon Greedy Algorithm
One challenge with the Q-function above is choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the Q-values observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? This is known as the exploration vs. exploitation dilemma.

Furthermore, the value of epsilon is purposely decayed over time, so that the agent favors exploration during its initial interactions with the environment, but increasingly favors exploitation as it gains more experience. The starting and ending values for epsilon, and the rate at which it decays are three hyperparameters that are later tuned during experimentation.

You can find the 𝛆-greedy logic implemented as part of the agent.act() method here in dqn_agent.py of the source code.

## Deep Q-Network (DQN)
With Deep Q-Learning, a deep neural network is used to approximate the Q-function. Given a network F, finding an optimal policy is a matter of finding the best weights w such that F(s,a,w) ≈ Q(s,a).

As for the network inputs, rather than feeding-in sequential batches of experience tuples, I randomly sample from a history of experiences using an approach called Experience Replay.

## Experience Replay
Experience replay allows the RL agent to learn from past experience.
In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. To replay important transitions more frequently, and therefore learn more efficiently, we use prioritized Experience Replay
The implementation of the replay buffer can be found here in the dqn_agent.py file of the source code.

## Duelling Agents
Dueling networks utilize two streams: one that estimates the state value function V(s), and another that estimates the advantage for each action A(s,a). These two values are then combined to obtain the desired Q-values.
