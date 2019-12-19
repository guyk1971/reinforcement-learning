# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 12:32:03 2016
MCCtrl_off_wis.py
Off-Policy MC Control with Weighted Importance Sampling.ipynb
@author: guy
"""
#%matplotlib inline

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
def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

#%%
def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    
    def policy_fn(observation):
        nA=Q[observation].size   # get the number of possible actions in state 'observation'
        best_action=np.argmax(Q[observation])
        probA=np.eye(nA)[best_action] # set the probability of best action to 1.0 and all the rest 0.0
        return probA

    return policy_fn
#%%
def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.

    Strictly following the algorithm from Sutton's book section 5.7 (page 119):
    "Off-policy every-visit MC control "
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Lambda discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    
    every_visit_update = True
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # C is the denominator of the importance sampling weights    
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    for k_episode in range(num_episodes):   # external loop on episodes
        # Print out which episode we're on, useful for debugging.
        if k_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(k_episode, num_episodes), end="")
            sys.stdout.flush()

        # we're following the routine described in Lecture 5, slide 19: each step of evaluation is based on single episode        
        # sample an episode using policy
        episode=[]
        state=env.reset()
        for t in range(100):    # steps within the episode
            action_probs=behavior_policy(state) # get p(a/s). should be an array of |A| (=env.nA)
            action=np.random.choice(np.arange(len(action_probs)), p=action_probs)     # sample from the action distribution (P(a|s))
            next_state, reward, done, _ = env.step(action) # perform one step
            episode.append((state,action,reward)) # each step in the episode is comprised from a tuple (S[t],A[t],R[t+1])
            if done:
                break
            state = next_state

        # now we have the episode as a list of tupples
        G = 0.0
        W = 1.0  # importance weighting ratios
        if every_visit_update:
            #############################################
            # every visit update:
            # every time we'll visit (state,action), we'll update Q(s,a)
            for t in range(len(episode))[::-1]: # running from t=T-1 downto t=0
                # calculate Gt
                S,A,R=episode[t]
                G=discount_factor*G+R
                C[S][A]+=W
                Q[S][A]+=(W/C[S][A])*(G-Q[S][A]) # performing incremental update
                TargetAction=np.argmax(target_policy(S))
                if A != TargetAction:
                    break;
                
                W *= 1.0/behavior_policy(S)[A] # note : target policy is deterministic greedy thus target_policy[S][A]=1.0
        # else:  # first time visit - TODO
    return Q, target_policy
#%%
if __name__=='__main__':
    random_policy = create_random_policy(env.action_space.n)    # it has to be soft. i.e. >0 wherever target policy is >0
    Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)
    
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")    
