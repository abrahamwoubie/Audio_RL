import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent
from Environment import Environment
from Environment import *

# Settings
env = Environment(nRow, nCol)
agent = Agent(env)

number_of_iterations_per_episode=[]
number_of_episodes=[]

# Train agent
print("\nTraining agent...\n")
N_episodes =2000
reward_List=[]
for episode in range(N_episodes):

    # Generate an episode
    reward_episode = 0

    state = env.reset()  # starting state
    #while True:
    number_of_episodes.append(episode)
    iteration=0
    while iteration < 100:
        iteration+=1
        action = agent.get_action(env)  # get action
        state_next, reward, done = env.step(action)  # evolve state by action
        agent.train((state, action, state_next, reward, done))  # train agent
        reward_episode += reward
        if done:
            break
        state = state_next  # transition to next state

    # Decay agent exploration parameter
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

    # Print
    #if (episode == 0) or (episode + 1) % 10 == 0:
    number_of_iterations_per_episode.append(iteration)
    reward_List.append(reward)

    print("[episode {}/{}] Number of Iterations = {}, Reward per episode = {}".format(
            episode + 1, N_episodes, iteration, reward_episode))

    # Print greedy policy
    if (episode == N_episodes - 1):
        agent.display_greedy_policy()
        # for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        #     print(" action['{}'] = {}".format(key, val))
        # print()
percentage_of_successful_episodes=(sum(reward_List)/N_episodes)*100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes,'%'))
fig = plt.figure()
fig.suptitle('Q-Learning', fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
# plt.grid(True)
#plt.savefig("Q_Learning_10_10.png")
plt.show()
