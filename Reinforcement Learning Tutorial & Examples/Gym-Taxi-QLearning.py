import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env =  gym.make("Taxi-v2")

# Q Table
q_table = np.zeros([env.observation_space.n,env.action_space.n])

# Hyperparameter
alpha = 0.1
gamma = 0.97
epsilon = 0.1

# Plotting metrics
reward_list = []
dropoff_list = []

episode = 10000

for i in range(1,episode):

    # initialize env
    state = env.reset()

    reward_count = 0
    dropoff = 0

    while True:

        # Exploit vs Explorer , find action %10 explore , %90 exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action , observe state and reward
        next_state, reward, done, _  = env.step(action)

        # old value
        old_value = q_table[state,action]

        # next max
        next_max = np.max(q_table[next_state])
    
        # Q Learning function
        next_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)

        # Q Table update
        q_table[state,action] = next_value 

        # Update State
        state = next_state

        # Wrong drop-off
        if reward == -10:
            dropoff += 1

        if done:
            break

        reward_count += reward
    if i % 10 == 0: 
        dropoff_list.append(dropoff)
        reward_list.append(reward_count)
        print("Episode: {}, Reward: {}, Wrong Drop: {}".format(i,reward_count,dropoff))


# Visualize

fig, axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropoff_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropoff")

axs[0].grid(True)
axs[1].grid(True)

plt.show()