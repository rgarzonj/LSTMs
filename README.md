# LSTMs
This repository includes the tests with the Blocksworld environment and LSTMs

Tests with LSTMs

## Blocksworld environment on OpenAI Gym
I implemented the interface for OpenAI Gym environments for the BlocksWorld environment based on the work of [Slaney and Thiébaux] (https://users.cecs.anu.edu.au/~jks/bw.html)

Follow these steps in order to configure the environment:
- Identify your gym installation by running 
python
import gym
print (gym.__file__)
- Copy the file blocksworld.py to the folder gym/envs/classic_control
- Edit the file gym/envs/classic_control/__init__.py and add the following line:
from gym.envs.classic_control.blocksworld import BlocksWorldEnv

- Download the file GENERATOR and compile it
- Edit the file blocksworld.py and configure the path to the binary file bwstates 


##


How to visualize the tensorflow learning
tensorboard --logdir=./BlocksWorld-v0/
http://localhost:6006


List of files

