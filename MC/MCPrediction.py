# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:44:56 2016
MCPrediction.py
@author: guy
"""
# %matplotlib inline

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()
#%%

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # The final value function
    V = defaultdict(float)
    # the model state is represented by a 3-tupple : (score, dealer_score, usable_ace) and this will be the key to the dictionaries
    for i_episode in range(num_episodes):
        # print out episode number every 1000 episodes:
        if i_episode % 1000 ==0:
            print("\rEpisode {}/{}".format(i_episode+1,num_episodes),end="")
            sys.stdout.flush()
        # now generate an episode as an array of tupples (score, dealer_score, usable_ace)
        episode=[]
        state=env.reset() # this is t=0. we get S0,R0 
        for t in range(100): # timesteps within the episode
            # each time step is splitted to 2. the time tick happens between these:
            #  1. the agent is doing an action A[t]
            #  2. the environment is responding with reward R[t+1] and moves to state S[t+1]
            action_probs=policy(state) # get p(a/s). should be an array of |A| (=env.nA)
            action=np.random.choice(np.arange(len(action_probs)), p=action_probs)     # sample from the action distribution (P(a|s))
            next_state, reward, done, _ = env.step(action) # perform one step
            episode.append((state,action,reward)) # each step in the episode is comprised from a tuple (S[t],A[t],R[t+1])
            if done:
                break
            state = next_state
        # now we have the episode in the array (of tupples) where each tupple is a step (S[t],A[t],R[t+1])
        # there are 2 possible options : (following the expression in Lecture 4 slide 13)
        # doing 'first visit update' or 'every visit update' ? 
        states_visited_list=[x[0] for x in episode]
        T=len(states_visited_list)  # the length of the episode
        #############################################
        # first visit update:
        # first create a list of the states visited in this episode
#        states_visited_set=set(states_visited_list)      # create a unique list of state (i.e. each state appears only once)
        # next, for each state find its first occurence , and add G to its accumulator
#        for state in states_visited_set:
#            step_first_visit=states_visited_list.index(state)    # find the step index of the first visit
#            # calculate G
#            discount_power=np.arange(0,T-step_first_visit) # the power of discount factor to multiply the rewards (from 0 to T-(t+1) )
#            disc_coef=discount_factor ** discount_power
#            _,_,rewards=zip(*episode[step_first_visit:]) # extract the rewards from time step_first_visit to T
#            G=np.sum(rewards*disc_coef) # multiply by discount factor
#            returns_count[state] += 1   # increment counter
#            returns_sum[state] += G     # add the total reward of this episode
            
        #############################################
        # every visit update:
        for t,state in enumerate(states_visited_list):
            # calculate G
            discount_power=np.arange(0,T-t) # the power of discount factor to multiply the rewards (from 0 to T-(t+1) )
            disc_coef=discount_factor ** discount_power
            _,_,rewards=zip(*episode[t:]) # extract the rewards from time step_first_visit to T
            G=np.sum(rewards*disc_coef)
            returns_count[state] += 1   # increment counter
            returns_sum[state] += G     # add the total reward of this episode
            
            
    # at this stage we have the aggregated the total returns over all episodes in returns_sum (S) and the counters in return_count (N)
    # all is left to do is to calculate V by doing S/N:
    for state in returns_count.keys():
        V[state]=returns_sum[state]/returns_count[state]
    
    return V

#%%
def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])

#%%
if __name__=='__main__':
#    V_10 = mc_prediction(sample_policy, env, num_episodes=10)
    
    V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
    plotting.plot_value_function(V_10k, title="10,000 Steps")
    
    V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
    plotting.plot_value_function(V_500k, title="500,000 Steps")

    
