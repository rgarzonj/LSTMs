# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys

sys.path.append("/Users/rubengarzon/Documents/Projects/phD/Repo/gym")

import random
import gym


numBlocks = 3
env = gym.make('BlocksWorld-v0')

env.seed(0)
env.reset()


done = False
while (done == False):
    next_action = [random.randint(0,numBlocks-1),random.randint(0,numBlocks)]    
    obs, reward, done, empty = env.step (next_action)    
    print ('Next action ' + str(next_action))
    print ('Obs ' + str(obs))
    env.render()
    
    input("Press Enter to continue...")

#env.reset()
#env.render()