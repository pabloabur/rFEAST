import numpy as np
import matplotlib.pyplot as plt
import random
import math

patterns = np.zeros((5, 3, 3))
patterns[0, 1, :] = 1
patterns[1, :, 1] = 1
patterns[2, :, :] = np.eye(3)
patterns[3, :, :] = np.flip(np.eye(3), 1)
patterns[4, :, :] = np.random.randint(2, size=(3, 3))

t = 0
#state_patterns = [[0, 1], [2, 3]]
#states = 2
# Alternatively
state_patterns = [[0, 1, 2, 3]]
states = 1
observations = [[] for x in range(states)]
num_patterns = 30

# Creating observations from states
for state in range(states):
    for _ in range(num_patterns):
        t += 1
        pattern = random.choice(state_patterns[state])
        for j in range(0, 3):
            for k in range(0, 3):
                if patterns[pattern, j, k] == 1:
                    observations[state].append((j, k, t))

""" Creating feast """
num_neurons = 4
input_dim = 9

# Initialization
w = np.random.rand(num_neurons, input_dim)
thres = np.zeros((num_neurons, 1))
thres_open = .01
thres_close = .001
w_norm = np.linalg.norm(w, axis=1)
w_norm = w_norm[:, np.newaxis]  # Increase dimension for division
w = w/w_norm
event_tau = 2
eta = .01
T = np.zeros((3, 3))  # Time surface

# Run throught events
for epoch in range(50):
    for index, sample in enumerate(observations):
        # In case we want to skip learning of a state
        #if index == 0:
        #    continue
        for event in sample:
            (x, y, ts) = event
            T[x, y] = ts*10  # space events so that a single event is learned
            event_context = np.exp((T-ts)/event_tau).reshape(9, 1)
            event_context = event_context/np.linalg.norm(event_context)
            print(event_context)
            dist = np.dot(w, event_context)  # cos(theta) = A*B/||A||/||B||
            print(dist)
            dist[dist < thres] = 0
            winner = np.argmax(dist)  # Looking for when theta=0 => cos(theta)=1
            if dist[winner] == 0:  # When no one reaches threshold
                thres = thres - thres_open
            else:
                w[winner, :] = (1-eta)*w[winner, :] + eta*event_context.T
                w[winner, :] = w[winner, :]/np.linalg.norm(w[winner, :])
                thres[winner] = thres[winner] + thres_close

# Plots
sq_n_neurons = math.ceil(math.sqrt(num_neurons))
fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, :], (3, 3)))
    axs.flat[i].set_title(str(thres[i]))

for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, :], (3, 3)))
    axs.flat[i].set_title(str(thres[i]))
plt.show()
