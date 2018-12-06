import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt

from Train_Agent import Agent



"""

 gridworld.py 

 We use Q-learning to train an epsilon-greedy agent to find the shortest path 
 between position (0, 0) to opposing corner (Ny-1, Nx-1) of a 2D rectangular grid
 in the 2D GridWorld environment of size (Ny, Nx).

 Note: 
 The optimal policy exists but is a highly degenerate solution because
 of the multitude of ways one can traverse down the grid in the minimum
 number of steps. Therefore a greedy policy that always moves the agent closer 
 towards the goal can be considered an optimal policy (can get to the goal 
 in `Ny + Nx - 2` actions). In our example, this corresponds to actions 
 of moving right or down to the bottom-right corner.

 Example optimal policy:
 
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

class Env:
    
    def __init__(self, Ny=8, Nx=8):
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        # Define rewards table
        self.R = self._build_rewards()  # R(s,a) agent rewards
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (0, 0)  
        return self.state

    def step(self, action):
        # Evolve agent state
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        # Collect reward
        reward = self.R[self.state + (action,)]
        # Terminate if we reach bottom-right grid corner
        done = (state_next[0] == self.Ny - 1) and (state_next[1] == self.Nx - 1)
        # Update state
        self.state = state_next
        return state_next, reward, done
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def _build_rewards(self):
        # Define agent rewards R[s,a]
        r_goal = 1  # reward for arriving at terminal state (bottom-right corner)
        r_nongoal = 0  # penalty for not reaching terminal state
        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
        R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = r_goal  # arrive from above
        R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = r_goal  # arrive from the left
        return R
