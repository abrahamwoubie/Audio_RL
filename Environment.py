import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent

#from Extract import Extract_Samples

from scipy.spatial import  distance


"""

 Example
 
  [[1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 0]]

  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3 

"""

nRow=4
nCol=4

class Environment:
    
    def __init__(self, nRow, nCol):
        # Define state space
        self.nRow = nRow  # x grid size
        self.nCol = nCol  # y grid size
        self.state_dim = (nRow, nCol)
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        # Define rewards table
        #self.R = self._build_rewards()  # R(s,a) agent rewards
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")


    def reset(self):
        # Reset agent state to top-left grid corner
        start_row=random.choice(range(0,nRow-1))
        start_col=random.choice(range(0,nCol-1))
        self.state = (start_row, start_col)

        goal_row = random.choice(range(0, nRow - 1))
        goal_col = random.choice(range(0, nCol - 1))
        self.goal_state=(goal_row,goal_col)

        return self.state,self.goal_state



    def step(self, action):
        # Evolve agent state
        reward = 0
        done = False
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])

        # samples_state_next=Extract_Samples(state_next[0],state_next[1])
        # samples_goal_state = Extract_Samples(nRow - 1, nCol - 1)

        #print("Extracted Samples of row {} col {} is {}".format(state_next[0],state_next[1],samples_state_next))

        # Collect reward
        if(state_next==(self.nRow-1,self.nCol-1)):
        #if(state_next==self.goal_state):
            reward=1
            done=True

        #reward = self.R[self.state + (action,)] #  state(0,0) and action (1,)=>(0,0,1)
        # if(distance.euclidean(samples_state_next,samples_goal_state)==0):
        #     reward=1
        #     done=True
        # else:
        #     reward=0
        #     done=False

        # Terminate if we reach bottom-r
        # right grid corner
        #done = (state_next[0] == self.nRow - 1) and (state_next[1] == self.nCol - 1)
        # Update state
        self.state = state_next
        #print("Returned next state",state_next)
        return state_next, reward, done
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        row, col = self.state[0], self.state[1]
        #print("y",self.state[0])
        #print("x", self.state[1])
        if (row > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (row < self.nRow - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (col > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (col < self.nCol - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        #print("actions allowed",actions_allowed)
        return actions_allowed

    # def _build_rewards(self):
    #     # Define agent rewards R[s,a]
    #     reward_goal = 1  # reward for arriving at terminal state (bottom-right corner)
    #     reward_no_goal = 0  # penalty for not reaching terminal state
    #     R = reward_no_goal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
    #     R[self.nRow - 2, self.nCol - 1, self.action_dict["down"]] = reward_goal  # arrive from above
    #     R[self.nRow - 1, self.nCol - 2, self.action_dict["right"]] = reward_goal  # arrive from the left
    #     return R

def Extract_Samples(row, col):

    fs = 100  # sample rate
    f = 2  # the frequency of the signal

    x = np.arange(fs)  # the points on the x axis for plotting

    # compute the value (amplitude) of the sin wave at the for each sample
    # if letter in b'G':
    if(row==nRow and col==nCol):
        samples = [100 + row + col + np.sin(2 * np.pi * f * (i / fs)) for i in x]
    else:
        samples = [row + col + np.sin(2 * np.pi * f * (i / fs)) for i in x]
    return samples