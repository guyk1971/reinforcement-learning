# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 12:44:59 2016
PolicyIteration.py
@author: guy
"""
import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

#%%
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    finished=False
    while not finished:
        Vprev=V
        V = np.zeros(env.nS)
        for s in range(env.nS):     # for each state
            for a in range(env.nA):     # for each possible action (assuming impossible actions are with P[s][a]=0)
                for P_a_ss, s_next_id, R_sa, done in env.P[s][a]:
                    V[s]+=policy[s,a]*P_a_ss*(R_sa+discount_factor*Vprev[s_next_id])
        finished= max(np.abs(V-Vprev))<theta
    return np.array(V)
#%%
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    finished = False
    V=policy_eval(policy,env,discount_factor) # get the value of the current policy
    while not finished:
        # TODO: Implement this!
        # now update each policy[s,a] in a greedy manner based on V
        for s in range(env.nS):     # for each state
            qs=np.zeros(env.nA)
            for a in range(env.nA):  # and for each action in this state
                # find the action that will maximize q(s,a)
                # if we assume deterministic policy then policy[s,a_max]=1.0 where all other elements in policy[s] are 0.
                # so we need to find a_max
                # a_max = argmax qs[a] so we need to calculate it here and at the end of the loop search for the max value
                for P_a_ss, s_next_id, R_sa, done in env.P[s][a]:
                    qs[a]+=P_a_ss*(R_sa+discount_factor*V[s_next_id])
            a_max=np.argmax(qs)
            policy[s]=np.zeros(env.nA)
            policy[s,a_max]=1.0
        # check stopping criteria
        Vprev=V
        V=policy_eval(policy,env,discount_factor) # get the value of the current policy
        finished= max(np.abs(V-Vprev))<0.001
    return policy, V

#%%
if __name__=='__main__':
    policy, v = policy_improvement(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")
    
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")
    
    print("Value Function:")
    print(v)
    print("")
    
    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    # Test the value function
    expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
