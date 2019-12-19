# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:03:30 2016
MCctrl_egreedy.py
MC Control with Epsilon-Greedy Policies
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
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        probA=epsilon*np.ones(nA,dtype=float)/nA
        probA[np.argmax(Q[observation])]+=(1.0-epsilon)
        return probA
    return policy_fn
    
#%%
def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """
    every_visit_update=True
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    for k_episode in range(num_episodes):   # external loop in episode
        # Print out which episode we're on, useful for debugging.
        if k_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(k_episode, num_episodes), end="")
            sys.stdout.flush()

        # we're following the routine described in Lecture 5, slide 19: each step of evaluation is based on single episode        
        # sample an episode using policy
        episode=[]
        state=env.reset()
        for t in range(100):    # steps within the episode
            action_probs=policy(state) # get p(a/s). should be an array of |A| (=env.nA)
            action=np.random.choice(np.arange(len(action_probs)), p=action_probs)     # sample from the action distribution (P(a|s))
            next_state, reward, done, _ = env.step(action) # perform one step
            episode.append((state,action,reward)) # each step in the episode is comprised from a tuple (S[t],A[t],R[t+1])
            if done:
                break
            state = next_state
        # now we have the episode as a list of tupples
        T=len(episode)
        state_action_visited=[(x[0],x[1]) for x in episode] # create a list of tupples (state,action) in the episode
        sa_pairs_visited_set=set(state_action_visited)      # create a unique list of state,action pairs

        if every_visit_update:
            #############################################
            # every visit update:
            # every time we'll visit (state,action), we'll update Q(s,a)
            for t,sa_pair in enumerate(state_action_visited):
                # calculate G
                discount_power=np.arange(0,T-t) # the power of discount factor to multiply the rewards (from 0 to T-(t+1) )
                disc_coef=discount_factor ** discount_power
                _,_,rewards=zip(*episode[t:]) # extract the rewards from time step_first_visit to T
                G=np.sum(rewards*disc_coef)
                returns_count[sa_pair]+=1
                returns_sum[sa_pair]+=G

        else: # first visit update
            #############################################
            # first visit update:
            # for each state_action find its first occurence , and add G to its accumulator
            for sa_pair in sa_pairs_visited_set:
                step_first_visit=state_action_visited.index(sa_pair)    # find the step index of the first visit
                # calculate G
                discount_power=np.arange(0,T-step_first_visit) # the power of discount factor to multiply the rewards (from 0 to T-(t+1) )
                disc_coef=discount_factor ** discount_power
                _,_,rewards=zip(*episode[step_first_visit:]) # extract the rewards from time step_first_visit to T
                G=np.sum(rewards*disc_coef) # multiply by discount factor
                returns_count[sa_pair] += 1   # increment counter
                returns_sum[sa_pair] += G     # add the total reward of this episode
        
        # now calculate Q from returns_sum and returns_count:
        for state,action in sa_pairs_visited_set:
            sa_pair=(state,action)
            Q[state][action]=returns_sum[sa_pair]/returns_count[sa_pair]
        
    # The policy is improved implicitly by changing the Q dictionary
    return Q, policy
#%%
def mc_control_epsilon_greedy_ref(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2] for x in episode[first_occurence_idx:]])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        # The policy is improved implicitly by changing the Q dictionar
    
    return Q, policy    
#%%
if __name__=='__main__':
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
    
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")


