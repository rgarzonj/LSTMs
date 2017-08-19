# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys

sys.path.append("/Users/rubengarzon/Documents/Projects/phD/Repo/gym")

import random
import gym


numBlocks = 2
env = gym.make('BlocksWorld-v0')

env.seed(0)
env.reset()


done = False
num_episodes = 1000
ep_lengths = []
n = 0
while (n<num_episodes):    
    steps =1
    done = False
    while (done == False):
        next_action = [random.randint(0,numBlocks-1),random.randint(0,numBlocks)]    
        obs, reward, done, empty = env.step (next_action)    
        print ('Next action ' + str(next_action))
        print ('Obs ' + str(obs))
        #env.render()
        steps +=1    
    ep_lengths.append(steps)
    n+=1

print ("Average episode length " + str(sum(ep_lengths) / float(len(ep_lengths))))
    #input("Press Enter to continue...")

#env.reset()
#env.render()