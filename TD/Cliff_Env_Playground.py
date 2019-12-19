# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:16:40 2016
Cliff_Env_Playground.py
@author: guy
"""
import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.cliff_walking import CliffWalkingEnv
env = CliffWalkingEnv()

#%%

print(env.reset())
env.render()

print(env.step(0))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()