import os
import gym
import csv
import pickle
import random
import numpy as np
from collections import namedtuple, deque


env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


###################################################################
# Part 1: Randomly Generate Samples from "Pendulum-v1"
###################################################################
num_envs = 20
num_episodes = 100
MEMORY_SIZE = num_envs*num_episodes

dist = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]    # cardinality = 10
var_samples = random.choices([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dist, k=num_envs)
L = [np.array((var)) for var in zip(var_samples)]

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
memory = deque(maxlen=MEMORY_SIZE)

def collect_individual_samples(env, index, num_episodes):
    done = False
    state = env.reset()
    memory = []
    for _ in range(num_episodes):
        action = env.action_space.sample() 
        next_state, reward, done, _ = env.step(action)
        memory.append((index, state, action, reward, next_state, done))
        state = next_state
    return memory

with open(os.path.join("..", "control/dataset", "pendulum.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user", "l", "s", "a", "s'"])

    for env in range(num_envs):
        gravity = L[env].item()
        random_samples = collect_individual_samples(gym.make("Pendulum-v1",g=gravity), env, num_episodes)

        # Populate memory with random samples
        for sample in random_samples:
            memory.append(sample)
            idx = sample[0]
            state = sample[1]
            action = sample[2]
            next_state = sample[4]
            augmented = (env, L[env].item(), state, action, next_state)
            writer.writerow(augmented)

with open(os.path.join("..", "control/dataset", "data_offline_multiple.pkl"), 'wb') as file:
    pickle.dump(memory, file)
