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
wi = np.random.rand(num_neurons, input_channels * input_channels)
we = np.random.rand(num_neurons, num_neurons)
thres = np.zeros((num_neurons, 1))
thres_open = .01
thres_close = .001
w_norm = np.linalg.norm(wi, axis=1)
w_norm = w_norm[:, np.newaxis]  # Increase dimension for division
wi = wi/w_norm
w_norm = np.linalg.norm(we, axis=1)
w_norm = w_norm[:, np.newaxis]
we = we/w_norm
event_tau = 2
eta = .01
T = np.zeros((input_channels, input_channels))  # Time surface

#
duration = []
prev_world_t = 0
feature_winner = {'w1': [], 'w2': []}

# Run throught events
# TODO reset world
for epoch in range(50000):
    t += 1
    random_walk += np.random.normal(0, .1)
    if random_walk > 1 or random_walk < 0:
        #import pdb;pdb.set_trace()
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

    # Process feedforward input
    event_context = T.reshape(input_channels*input_channels, 1)
    event_context = event_context/np.linalg.norm(event_context)
    dist = np.dot(wi, event_context)  # cos(theta) = A*B/||A||/||B||
    dist[dist < thres] = 0
    winner = np.argmax(dist)  # Looking for when theta=0 => cos(theta)=1
    if dist[winner] == 0:  # When no one reaches threshold
        thres = thres - thres_open
        #continue
    else:
        wi[winner, :] = (1-eta)*wi[winner, :] + eta*event_context.T
        wi[winner, :] = wi[winner, :]/np.linalg.norm(wi[winner, :])
        thres[winner] = thres[winner] + thres_close
        if world == 1:
            feature_winner['w1'].append(winner)
        else:
            feature_winner['w2'].append(winner)
    # TODO generate event context from winner?

    # Process recurrent input
    # TODO associate winners so that it is seen on recurrent matrix

# TODO dot product wi*ei+wr*er?

# Plots
sq_n_neurons = math.ceil(math.sqrt(num_neurons))
fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(wi[i, :], (input_channels, input_channels)))
    axs.flat[i].set_title(str(thres[i]))

plt.figure()
_=plt.hist(feature_winner['w1'], bins=num_neurons, range=(0, num_neurons-1), color='k', histtype='step')
_=plt.hist(feature_winner['w2'], bins=num_neurons, range=(0, num_neurons-1), histtype='step')

plt.show()
