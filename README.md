# LSTMs
This repository includes the tests with the Blocksworld environment and LSTMs
Based on the code of DQN from dennybritz

## Blocksworld environment on OpenAI Gym
I implemented the interface for OpenAI Gym environments for the BlocksWorld environment based on the work of [Slaney and Thiébaux] (https://users.cecs.anu.edu.au/~jks/bw.html)

Follow these steps in order to configure the environment:
- Identify your gym installation by running 
```python
import gym
print (gym.__file__)
```
- Copy the file /Blocksworld/classic_control/blocksworld.py to the folder gym/envs/classic_control
- Edit the file gym/envs/__init__.py and add the lines
```python
register(
    id='BlocksWorld-v0',
    entry_point='gym.envs.classic_control:BlocksWorldEnv',
    max_episode_steps=20000,
)
```
- Edit the file gym/envs/classic_control/__init__.py and add the following line:
```python
from gym.envs.classic_control.blocksworld import BlocksWorldEnv
```

- Go to the folder Blocksworld/GENERATOR/bwstates.1 and compile the binary 'bwstates'. Alternatively download the file bwstates.1.tar.gz and compile for your platform.
- Edit the file blocksworld.py and configure the path to the binary file 'bwstates' 


### Visualize tensorflow learning
tensorboard --logdir=./BlocksWorld-v0/
http://localhost:6006


### About cell states, hidden states and outputs
The naming conventions are quite confusing in the LSTM implementations I found.

As per tensorflow reference of the BasicLSTMCell
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell

I've followed the conventions in 
http://arxiv.org/abs/1409.2329

Basically:
- The cell state is the same as the hidden state (named c)
- The outputs are named with h


### How to use the LSTMVis module

See https://github.com/HendrikStrobelt/LSTMVis

Once the file DQN_LSTM_BlocksWorld.py has completed all the episodes, a folder called /lstmvis will be created inside the 
experiment folder.

This folder contains the files
- lstm.yml
- states.hdf5
- train.dict
- train.hdf5


### Guide to the files in this repository

DQN_LSTM_BlocksWorld.py
blocksworld.py




