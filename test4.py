""" Here, differently from previous test, event contexts are normalized
    and processed separetely."""

import numpy as np
import matplotlib.pyplot as plt
import random

import math
import sys
import warnings
warnings.filterwarnings('error')

def plot_classifications(num_neurons, class_winners, feature_mappings, color_mappings):
    plt.figure()
    bar_width = 0.1
    bar_offset = -0.2
    for event_class, val in feature_mappings.items():
        plt.bar(np.arange(num_neurons) + bar_offset, class_winners[event_class],
                bar_width, label=feature_mappings[event_class],
                color=color_mappings[event_class])
        bar_offset += bar_width
    plt.xlabel('Neuron index')
    plt.ylabel('Number of wins')
    plt.legend()
    plt.pause(0.1)

def average_histograms(histogram_samples, num_neurons):
    avg_hist = {0: np.array([0 for _ in range(num_neurons)], dtype=float),
                1: np.array([0 for _ in range(num_neurons)], dtype=float),
                2: np.array([0 for _ in range(num_neurons)], dtype=float),
                3: np.array([0 for _ in range(num_neurons)], dtype=float),
                4: np.array([0 for _ in range(num_neurons)], dtype=float),
                5: np.array([0 for _ in range(num_neurons)], dtype=float)}

    for sample_hist in histogram_samples:
        for key in sample_hist.keys():
            avg_hist[key] += sample_hist[key]
    num_samples = len(histogram_samples)
    for key in avg_hist.keys():
        avg_hist[key] /= num_samples

    return avg_hist

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
eta = .01
recurrent_tau = 50

# Time surfaces
T = np.zeros((input_channels, input_channels))
rec_T = np.zeros((num_neurons, 1))

# Defines importance of feedforward (alpha) versus recurrent (beta) weights
alpha = 1.
beta = 1.0 - alpha

# Variables used to evaluate world transitions
duration = []
prev_world_t = 0
feature_class = np.nan
feature_mappings = {0: 'vertical', 1: 'horizontal',
                    2: 'desc_diagonal', 3: 'asc_diagonal',
                    4: 'noise_axis', 5: 'noise_diag'}
color_mappings = {0: 'blue', 1: 'orange',
                    2: 'green', 3: 'red',
                    4: 'purple', 5: 'yellow'}
class_winners = {0: [0 for _ in range(num_neurons)],
                 1: [0 for _ in range(num_neurons)],
                 2: [0 for _ in range(num_neurons)],
                 3: [0 for _ in range(num_neurons)],
                 4: [0 for _ in range(num_neurons)],
                 5: [0 for _ in range(num_neurons)]}
feature_t = []
winners_t = []
noise_flag = False
noise_responses = []
unmatched_noise = 0
classified_noise = []
noise_count = 0

# Run throught events
epochs = range(10)
iterations = 6500
monitor_class_winners = []
monitor_thres = np.zeros((num_neurons, len(epochs)*iterations+1))
for epoch in epochs:
    for iteration in range(iterations):
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

        #TODO better make another loop with learning turned off
        if np.random.rand() < .1:
            #T = np.random.randint(2, size=np.shape(T))
            #T = np.zeros(np.shape(T))
            T = np.random.rand(*np.shape(T))
            noise_flag ^= True
            noise_count += 1
            if feature_class == 0 or feature_class == 1:
                feature_class = 4
            else:
                feature_class = 5
        #if t==50001:
        #    sq_n_neurons = math.ceil(math.sqrt(num_neurons))
        #    fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
        #    plt.title('feedforward')
        #    for i in range(num_neurons):
        #        axs.flat[i].imshow(np.reshape(w[i, :input_dim], (input_channels, input_channels)))
        #        axs.flat[i].set_title(f'neu{i}, thr:{thres[i]}')
        #    if mode == 'rec':
        #        fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
        #        plt.title('recurrent')
        #        for i in range(num_neurons):
        #            axs.flat[i].imshow(np.reshape(w[i, input_dim:], (np.sqrt(num_neurons).astype(int), np.sqrt(num_neurons).astype(int))))
        #            axs.flat[i].set_title(f'neu{i}, thr:{thres[i]}')
        #    plt.show()

        # Join and normalize event contexts
        feedforward_event_context = T.reshape(input_dim, 1)
        feedforward_event_context /= np.linalg.norm(feedforward_event_context)
        recurrent_event_context = np.exp(rec_T-t) / recurrent_tau
        recurrent_event_context /= np.linalg.norm(recurrent_event_context)
        total_event_context = np.concatenate(
            (feedforward_event_context, recurrent_event_context), axis=0)

        # Defines winner
        # TODO normalize separately? If I do, no feature representation
        #if np.any(w[:, :input_dim]):
        #    w_ff = w[:, :input_dim] / np.linalg.norm(w[:, :input_dim])
        #else:
        #    w_ff = w[:, :input_dim]
        #if np.any(w[:, input_dim:]):
        #    w_rec = w[:, input_dim:] / np.linalg.norm(w[:, input_dim:])
        #else:
        #    w_rec = w[:, input_dim:]
        w_ff = w[:, :input_dim]
        w_rec = w[:, input_dim:]

        # TODO do I need weights here? Does not seem to make a diference
        #w = np.concatenate((alpha*w_ff, beta*w_rec), axis=1)
        w = np.concatenate((w_ff, w_rec), axis=1)
        # TODO do I need weights here?
        dist = (alpha * np.dot(w_ff, feedforward_event_context) # cos(theta) = A*B/||A||/||B||
              + beta * np.dot(w_rec, recurrent_event_context))
        dist[dist < thres] = 0
        winner = np.argmax(dist)  # Looking for when theta=0 => cos(theta)=1
        if dist[winner] == 0:  # When no one reaches threshold
            thres = thres - thres_open

            # Keep track of winners and states
            winners_t.append(-1)
            feature_t.append(feature_class)

            if noise_flag:
                unmatched_noise += 1
                noise_flag ^= True
        else:
            w[winner, :input_dim] = alpha * ((1-eta)*w[winner, :input_dim] + eta*feedforward_event_context.T)
            w[winner, input_dim:] = beta * ((1-eta)*w[winner, input_dim:] + eta*recurrent_event_context.T)
            # TODO Or all at once?
            #w[winner, :] = (1-eta)*w[winner, :] + eta*total_event_context.T
            w[winner, :] = w[winner, :]/np.linalg.norm(w[winner, :])
            thres[winner] = thres[winner] + thres_close
            rec_T[winner] = t

            # Keep track of winners and states
            winners_t.append(winner)
            feature_t.append(feature_class)
            class_winners[feature_class][winner] += 1

            if noise_flag:
                noise_responses.append(winner)
                classified_noise.append(T)
                noise_flag ^= True

        monitor_thres[:, t] = thres.flatten()

    monitor_class_winners.append(class_winners)
    #plot_classifications(num_neurons, class_winners, feature_mappings, color_mappings)
    class_winners = {0: [0 for _ in range(num_neurons)],
                     1: [0 for _ in range(num_neurons)],
                     2: [0 for _ in range(num_neurons)],
                     3: [0 for _ in range(num_neurons)],
                     4: [0 for _ in range(num_neurons)],
                     5: [0 for _ in range(num_neurons)]}

# Plots
sq_n_neurons = math.ceil(math.sqrt(num_neurons))
fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
plt.title('feedforward')
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, :input_dim], (input_channels, input_channels)))
    axs.flat[i].set_title(f'neu{i}, thr:{thres[i]}')

fig, axs = plt.subplots(sq_n_neurons, sq_n_neurons)
plt.title('recurrent')
for i in range(num_neurons):
    axs.flat[i].imshow(np.reshape(w[i, input_dim:], (np.sqrt(num_neurons).astype(int), np.sqrt(num_neurons).astype(int))))
    axs.flat[i].set_title(f'neu{i}, thr:{thres[i]}')

plt.figure()
_=plt.hist(noise_responses)
plt.xlabel('Neuron index')
plt.ylabel('# wins when noise was presented')

avg_hist = average_histograms(monitor_class_winners, num_neurons)
plot_classifications(num_neurons, avg_hist, feature_mappings, color_mappings)

plt.figure()
ax=plt.subplot(211)
plt.plot(feature_t)
plt.xlabel('Time (samples)')
plt.ylabel('Input class')
plt.yticks(np.arange(6), list(feature_mappings.values()), rotation=45)
plt.subplot(212, sharex=ax)
plt.plot(winners_t, '.')
plt.xlabel('Time (samples)')
plt.ylabel('Winner neuron index')
plt.hlines(0, 0, len(winners_t), linestyles='dashed', label='no winners')
plt.legend()

plt.figure()
for i in range(36):
    plt.plot(monitor_thres[i])
plt.xlabel('Time (samples)')
plt.ylabel('Thresholds')

print(f'noise occurences: {noise_count}')
print(f'Number of responses to noise: {len(noise_responses)}')
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
