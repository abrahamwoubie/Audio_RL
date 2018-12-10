import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, env, nRow, nCol):
        # Store state and action dimension
        #self.state_dim = env.state_dim
        #self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        #self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)
        self.Q=np.zeros([nRow*nCol,4])


    def get_action(self, env,nRow,nCol):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:

        # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0]*nRow+state[1], actions_allowed]
            #print("q[s]",self.Q[state[0], state[1], actions_allowed])
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        print("Memory",memory)
        #sa = state + (action,)
        #self.Q[sa] += self.beta * (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[sa])
        self.Q[state,action]+=self.beta * (reward + self.gamma * np.max(self.Q[state_next,:]) - self.Q[state,action])
        #Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        #print("Q[state,action]",self.Q[state,action])
        #print("Q[sa] {} is sa {} ".format(self.Q[sa],sa))

    # def display_greedy_policy(self):
    #     # greedy policy = argmax[a'] Q[s,a']
    #     greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
    #     print(self.Q)
    #     for x in range(self.state_dim[0]):
    #         for y in range(self.state_dim[1]):
    #             #print(self.Q[y, x, :])
    #             greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
    #     print("\nGreedy policy(y, x):")
    #     print(greedy_policy)
    #     print()
