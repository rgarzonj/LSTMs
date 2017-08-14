#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:44:11 2017

@author: rubengarzon
"""
#import os
#numBlocks = 3
#var = os.system('/Users/rubengarzon/Documents/Projects/phD/Repo/LSTMs/Blocksworld/GENERATOR/bwstates.1/bwstates -n ' + str(numBlocks))
#print (var)
#lines = var.split('\n')
#print (lines[1])

import subprocess
numBlocks = 3
seed = 1234
bwstates_path = '/Users/rubengarzon/Documents/Projects/phD/Repo/LSTMs/Blocksworld/GENERATOR/bwstates.1/bwstates -n ' + str(numBlocks) + ' -r ' + str(seed)
proc = subprocess.Popen(bwstates_path,stdout=subprocess.PIPE,shell=True)
(out, err) = proc.communicate()
#outwithoutreturn = out.rstrip('\n')
print (out)
out_str = out.decode('utf8')
lines = out_str.split('\n')
print (lines)
print (lines[1])

