import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

mode = sys.argv[1]

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
if mode == 'rec':
    w = np.random.rand(num_neurons, input_dim + num_neurons)
else:
    w = np.random.rand(num_neurons, input_dim)
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
if mode == 'rec':
    rec_T = np.zeros((num_neurons, 1))

# Variables used to evaluate world transitions
duration = []
prev_world_t = 0
feature_winner = {'w1': [], 'w2': []}
feature_class = np.nan
feature_mappings = {0: 'vertical', 1: 'horizontal',
                    2: 'desc_diagonal', 3: 'asc_diagonal',
                    4: 'noise'}
features = {0: 0, 1: 0, 2: 0, 3: 0}
class_winners = {0: [0 for _ in range(num_neurons)],
                 1: [0 for _ in range(num_neurons)],
                 2: [0 for _ in range(num_neurons)],
                 3: [0 for _ in range(num_neurons)],
                 4: [0 for _ in range(num_neurons)]}
feature_t = []
winners_t = []
noise_flag = False
false_positives = []
unmatched_noise = 0
classified_noise = []
noise_count = 0

# Run throught events
for epoch in range(65000):
    t += 1

    # Define world
    random_walk += np.random.normal(0, .1)
    if random_walk > 1 or random_walk < 0:
        world *= (-1)
        random_walk = .5
        duration.append(t - prev_world_t)
        prev_world_t = t

    # Generate samples on the fly
    if world == 1:
        k=np.random.randint(-input_channels+2, input_channels-2)
        T = np.eye(input_channels, k=k)
        for i in range(-thickness, thickness+1):
            if k+i < input_channels:
                T += np.eye(input_channels, k=k+i)
        feature_class = 2
        if random.uniform(0, 1) > .5:
            T = np.flip(T, axis=1)
            feature_class = 3
    else:
        T = np.zeros((input_channels, input_channels))  # Time surface
        k=np.random.randint(2, input_channels-2)
        T[:, k] = 1
        for i in range(-thickness, thickness+1):
            if k+i >= 0 and k+i < input_channels:
                T[:, k+i] += 1
        feature_class = 0
        if random.uniform(0, 1) > .5:
            T = T.T
            feature_class = 1

    if t>50000:
        #TODO better make another loop with learning turned off
        if np.random.rand() < .1:
            #T = np.random.randint(2, size=np.shape(T))
            #T = np.zeros(np.shape(T))
            T = np.random.rand(*np.shape(T))
            noise_flag ^= True
            noise_count += 1
            feature_class = 4

    # Join and normalize event contexts
    event_context = T.reshape(input_dim, 1)
    if mode == 'rec':
        recurrent_event_context = np.exp(rec_T-t) / recurrent_tau
        total_event_context = np.concatenate(
            (event_context, recurrent_event_context), axis=0)
    else:
        total_event_context = event_context
    total_event_context = total_event_context/np.linalg.norm(total_event_context)

    # Defines winner
    dist = np.dot(w, total_event_context)  # cos(theta) = A*B/||A||/||B||
    dist[dist < thres] = 0
    winner = np.argmax(dist)  # Looking for when theta=0 => cos(theta)=1
    if dist[winner] == 0:  # When no one reaches threshold
        thres = thres - thres_open
        winners_t.append(-1)
        if noise_flag:
            unmatched_noise += 1
            noise_flag ^= True
    else:
        w[winner, :] = (1-eta)*w[winner, :] + eta*total_event_context.T
        w[winner, :] = w[winner, :]/np.linalg.norm(w[winner, :])
        thres[winner] = thres[winner] + thres_close
        if mode == 'rec':
            rec_T[winner] = t

        # Keep track of winners and states
        winners_t.append(winner)
        feature_t.append(feature_class)
        class_winners[feature_class][winner] += 1
        if world == 1:
            feature_winner['w1'].append(winner)
        else:
            feature_winner['w2'].append(winner)

        if noise_flag:
            false_positives.append(winner)
            classified_noise.append(T)
            noise_flag ^= True


# Plots
sq_n_neurons = math.ceil(math.sqrt(num_neurons))
fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
plt.title('feedforward')
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, :input_dim], (input_channels, input_channels)))
    axs.flat[i].set_title(f'neu{i}, thr:{thres[i]}')


if mode == 'rec':
    fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
    plt.title('recurrent')
    for i in range(num_neurons):
        axs.flat[i].imshow(np.reshape(w[i, input_dim:], (np.sqrt(num_neurons).astype(int), np.sqrt(num_neurons).astype(int))))
        axs.flat[i].set_title(f'neu{i}, thr:{thres[i]}')

plt.figure()
_=plt.hist(feature_winner['w1'], bins=num_neurons, range=(0, num_neurons-1), color='k', histtype='step')
_=plt.hist(feature_winner['w2'], bins=num_neurons, range=(0, num_neurons-1), color='r', histtype='step')
plt.xlabel('Neuron index')
plt.ylabel('Wins')

plt.figure()
_=plt.hist(false_positives)
plt.xlabel('Neuron index')
plt.ylabel('Number of false positives')

plt.figure()
bar_width = 0.2
plt.bar(np.arange(num_neurons) - 0.4, class_winners[0], bar_width, label=feature_mappings[0])
plt.bar(np.arange(num_neurons) - 0.2, class_winners[1], bar_width, label=feature_mappings[1])
plt.bar(np.arange(num_neurons), class_winners[2], bar_width, label=feature_mappings[2])
plt.bar(np.arange(num_neurons) + 0.2, class_winners[3], bar_width, label=feature_mappings[3])
plt.bar(np.arange(num_neurons) + 0.4, class_winners[4], bar_width, label=feature_mappings[4])
plt.xlabel('Neuron index')
plt.ylabel('Number of wins')
plt.legend()

plt.figure()
ax=plt.subplot(211)
plt.plot(feature_t)
plt.xlabel('Time (samples)')
plt.ylabel('Class number')
plt.subplot(212, sharex=ax)
plt.plot(winners_t, '.')
plt.xlabel('Time (samples)')
plt.ylabel('Winner neuron index')

print(f'noise occurences: {noise_count}')
print(f'Number of false positives: {len(false_positives)}')
print(f'Number times no neurons responded to noise: {unmatched_noise}')

#plt.figure()
#plt.imshow(classified_noise[0])
#plt.figure()
#plt.imshow(classified_noise[10])
#plt.figure()
#plt.imshow(classified_noise[20])
#plt.figure()
#plt.imshow(classified_noise[30])

plt.pause(0.1)
