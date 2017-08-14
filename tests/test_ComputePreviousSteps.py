#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:34:52 2017

@author: rubengarzon
"""

from collections import deque, namedtuple
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])



replay_memory = []
state = 1
action = 2
reward = 3
next_state = 4
done = 5

replay_memory.append(Transition(state, action, reward, next_state, done))
replay_memory.append(Transition(state*2, action*2, reward*2, next_state*2, done*2))
replay_memory.append(Transition(state, action, reward, next_state, done))
replay_memory.append(Transition(state*2, action*2, reward*2, next_state*2, done*2))
replay_memory.append(Transition(state, action, reward, next_state, done))
replay_memory.append(Transition(state*2, action*2, reward*2, next_state*2, done*2))      



def computePreviousStates(replay_memory,seq_length,computeNextStates):
    prev_states = []
    
    if (len(replay_memory) > seq_length):
        n = seq_length    
    else:    
        n = len(replay_memory)    
    while (n > 0):
        if (computeNextStates == True):
            prev_states.append(replay_memory[-n].next_state)
        else:
            prev_states.append(replay_memory[-n].state)
        n = n - 1
    return prev_states

seq_length = 8

print (computePreviousStates(replay_memory,seq_length,True))    

seq_length = 2

print (computePreviousStates(replay_memory,seq_length,False))    
prev_states = computePreviousStates(replay_memory,seq_length,False)

# Add the state to the replay memory

Transition2 = namedtuple("Transition", ["state", "action", "reward", "next_state", "done","prev_states"])
replay_memory2 = []
replay_memory2.append(Transition2(state*2, action*2, reward*2, next_state*2, done*2,prev_states))    

print (replay_memory2)



    