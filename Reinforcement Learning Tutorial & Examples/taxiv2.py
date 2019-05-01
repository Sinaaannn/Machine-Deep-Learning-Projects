import gym

env = gym.make("Taxi-v2")

env.render()

# Blue = Passenger , Yellow = Empty Taxi, Purple = Destination, Green Full taxi , RGBY = location for destination and passenger

env.reset() # Reset env and return initial random initial state

print("State space: ", env.observation_space) # 500
print("Action space: ", env.action_space) # 6

# taxi row, taxi column, passenger index, destination
state = env.encode(3,1,2,2)

print("State number: ", state) # 330.state

env.s = state
env.render()

"""
For more info about env => https://github.com/openai/gym/blob/17abad35ae82cf01ff4dc962ffd63e3d849d3903/gym/envs/toy_text/taxi.py
Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger

{0: [(1.0, 430, -1, False)],
 1: [(1.0, 230, -1, False)],
 2: [(1.0, 350, -1, False)], 
 3: [(1.0, 330, -1, False)], 
 4: [(1.0, 330, -10, False)], 5: [(1.0, 330, -10, False)]}
"""
print(env.P[330])

# Lets get started one episode of game
time_step = 0
total_reward = 0
list_visualize = []
while True:

    time_step += 1

    # Choose action
    action = env.action_space.sample()

    # Perform action
    state, reward, done, _  = env.step(action) # state  => next state

    # Measure reward , total reward
    total_reward += reward

    # Visualize
    list_visualize.append({
        "frame" : env.render(mode = "ansi"),
        "state" : state,
        "action": action,
        "reward": reward,
        "total_reward": total_reward
    })

    env.render()

    if done:
        break


for i, frame in enumerate(list_visualize):
    print(frame.get("frame"))
    print("Time step: ", i+1)
    print("State: ", frame["state"])
    print("Action: ", frame["action"])
    print("Reward: ", frame["reward"])
    print("Total Reward: ", frame["total_reward"])