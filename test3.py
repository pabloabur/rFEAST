import numpy as np
import matplotlib.pyplot as plt
import random
import math

""" Creating feast """
num_neurons = 36
input_channels = 11
input_dim = input_channels * input_channels

# Creating contexts
t = 0
random_walk = .5
world = 1

# Initialization
thickness = 1
# Matrix containing both feedforward and recurrent weights
w = np.random.rand(num_neurons, input_dim + num_neurons)
thres = np.zeros((num_neurons, 1))
thres_open = .01
thres_close = .001
w_norm = np.linalg.norm(w, axis=1)
w_norm = w_norm[:, np.newaxis]  # Increase dimension for division
w = w/w_norm
w_norm = w_norm[:, np.newaxis]
eta = .01
recurrent_tau = 50

# Time surfaces
T = np.zeros((input_channels, input_channels))
rec_T = np.zeros((num_neurons, 1))

# Variables used to evaluate world transitions
duration = []
prev_world_t = 0
feature_winner = {'w1': [], 'w2': []}

# Run throught events
for epoch in range(65000):
    t += 1
    random_walk += np.random.normal(0, .1)
    if random_walk > 1 or random_walk < 0:
        world *= (-1)
        random_walk = .5
        duration.append(t - prev_world_t)
        prev_world_t = t
    if world == 1:
        k=np.random.randint(-input_channels+2, input_channels-2)
        T = np.eye(input_channels, k=k)
        for i in range(-thickness, thickness+1):
            if k+i < input_channels:
                T += np.eye(input_channels, k=k+i)
        if random.uniform(0, 1) > .5:
            T = np.flip(T, axis=1)
    else:
        T = np.zeros((input_channels, input_channels))  # Time surface
        k=np.random.randint(2, input_channels-2)
        T[:, k] = 1
        for i in range(-thickness, thickness+1):
            if k+i >= 0 and k+i < input_channels:
                T[:, k+i] += 1
        if random.uniform(0, 1) > .5:
            T = T.T

    if t>50000:
        #TODO better make another loop with learning turned off
        if np.random.rand() < .1:
            T = np.random.randint(2, size=np.shape(T))

    # Join and normalize event contexts
    event_context = T.reshape(input_dim, 1)
    recurrent_event_context = np.exp(rec_T-t) / recurrent_tau
    total_event_context = np.concatenate(
        (event_context, recurrent_event_context), axis=0)
    total_event_context = total_event_context/np.linalg.norm(total_event_context)

    # Defines winner
    dist = np.dot(w, total_event_context)  # cos(theta) = A*B/||A||/||B||
    dist[dist < thres] = 0
    winner = np.argmax(dist)  # Looking for when theta=0 => cos(theta)=1
    if dist[winner] == 0:  # When no one reaches threshold
        thres = thres - thres_open
    else:
        w[winner, :] = (1-eta)*w[winner, :] + eta*total_event_context.T
        w[winner, :] = w[winner, :]/np.linalg.norm(w[winner, :])
        thres[winner] = thres[winner] + thres_close

        rec_T[winner] = t

        # Keep track of winners
        if world == 1:
            feature_winner['w1'].append(winner)
        else:
            feature_winner['w2'].append(winner)


# Plots
sq_n_neurons = math.ceil(math.sqrt(num_neurons))
fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
plt.title('feedforward')
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, :input_dim], (input_channels, input_channels)))
    axs.flat[i].set_title(str(thres[i]))

fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
plt.title('recurrent')
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, input_dim:], (np.sqrt(num_neurons).astype(int), np.sqrt(num_neurons).astype(int))))
    axs.flat[i].set_title(str(thres[i]))

plt.figure()
_=plt.hist(feature_winner['w1'], bins=num_neurons, range=(0, num_neurons-1), color='k', histtype='step')
_=plt.hist(feature_winner['w2'], bins=num_neurons, range=(0, num_neurons-1), histtype='step')

plt.show()
