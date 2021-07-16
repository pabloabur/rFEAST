""" Here, differently from previous test, event contexts are normalized
    and processed separetely."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random

import math
import warnings
warnings.filterwarnings('error')

plt.ion()


class weights_monitor:
    def __init__(self, num_neurons, input_channels):
        self.num_neurons = num_neurons
        self.input_channels = input_channels
        self.sq_n_neurons = math.ceil(math.sqrt(self.num_neurons))

        self.fig_ff, self.axs_ff = plt.subplots(self.sq_n_neurons, self.sq_n_neurons)
        self.fig_ff.tight_layout(h_pad=0.2)
        self.im_ff = np.zeros_like(self.axs_ff)
        for i in range(self.num_neurons):
            self.im_ff.flat[i] = self.axs_ff.flat[i].imshow(
                np.random.rand(self.input_channels, self.input_channels),
                vmin=0, vmax=1)
            self.axs_ff.flat[i].set_title(f'neu{i}, thr:0.00', fontsize=10)

        self.fig_rec, self.axs_rec = plt.subplots(self.sq_n_neurons, self.sq_n_neurons)
        self.fig_rec.tight_layout(h_pad=0.2)
        self.im_rec = np.zeros_like(self.axs_rec)
        for i in range(self.num_neurons):
            self.im_rec.flat[i] = self.axs_rec.flat[i].imshow(
                np.random.rand(np.sqrt(self.num_neurons).astype(int),
                np.sqrt(self.num_neurons).astype(int)),
                vmin=0, vmax=1)
            self.axs_rec.flat[i].set_title(f'neu{i}, thr:0.00', fontsize=10)
        plt.pause(0.05)

    def update_plots(self, feedforward_w, recurrent_w, thres, autoscale=False):
        for i in range(self.num_neurons):
            weights = np.reshape(feedforward_w[i, :], (self.input_channels, self.input_channels))
            self.im_ff.flat[i].set_data(weights)
            self.axs_ff.flat[i].set_title(f'neu:{i}, thr:{thres[i]:.2f}', fontsize=10)
            if autoscale:
                self.im_ff.flat[i].autoscale()
        self.fig_ff.canvas.flush_events()

        for i in range(self.num_neurons):
            weights = np.reshape(recurrent_w[i, :], (np.sqrt(self.num_neurons).astype(int),
                                               np.sqrt(self.num_neurons).astype(int)))
            self.im_rec.flat[i].set_data(weights)
            self.axs_rec.flat[i].set_title(f'neu:{i}, thr:{thres[i]:.2f}', fontsize=10)
            if autoscale:
                self.im_rec.flat[i].autoscale()
        self.fig_rec.canvas.flush_events()


def plot_classifications(num_neurons, class_winners, feature_mappings, color_mappings):
    ax = plt.figure().gca()
    bar_width = 0.1
    bar_offset = -0.2
    for event_class, val in feature_mappings.items():
        plt.bar(np.arange(num_neurons) + bar_offset, class_winners[event_class],
                bar_width, label=feature_mappings[event_class],
                color=color_mappings[event_class])
        bar_offset += bar_width
    plt.xlabel('Neuron index', fontsize=24)
    plt.ylabel('Number of wins', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
num_neurons = 25
input_channels = 11
input_dim = input_channels * input_channels

# Creating contexts
t = 0
random_walk = .5
world = 1

# Initialization
thickness = 1
# Matrix containing both feedforward and recurrent weights
thres = np.zeros((num_neurons, 1))
thres_open = .001
thres_close = .001
w = np.random.rand(num_neurons, input_dim + num_neurons)
w_ff = w[:, :input_dim] / np.linalg.norm(w[:, :input_dim], axis=1, keepdims=True)
w_rec = w[:, input_dim:] / np.linalg.norm(w[:, input_dim:], axis=1, keepdims=True)
eta_ff = .01
eta_rec = .005
recurrent_tau = 50

# Time surfaces
T = np.zeros((input_channels, input_channels))
rec_T = np.zeros((num_neurons, 1))

# Defines importance of feedforward (alpha) versus recurrent (beta) weights
alpha = 0.65
beta = 1.0 - alpha

# Variables used to evaluate world transitions
feature_class = np.nan
feature_mappings = {0: 'vertical', 1: 'horizontal',
                    2: 'desc_diagonal', 3: 'asc_diagonal',
                    4: 'noise_2', 5: 'noise_1'}
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
classified_noise = []

# Prepare plots
w_mon = weights_monitor(num_neurons, input_channels)

# Run throught events
epochs = range(10)
iterations = 8000
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

        # Generate samples on the fly
        if world == 1:
            k = np.random.randint(-input_channels+2, input_channels-2)
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
            k = np.random.randint(2, input_channels-2)
            T[:, k] = 1
            for i in range(-thickness, thickness+1):
                if k+i >= 0 and k+i < input_channels:
                    T[:, k+i] += 1
            feature_class = 0
            if random.uniform(0, 1) > .5:
                T = T.T
                feature_class = 1

        if np.random.rand() < .1:
            T = np.random.rand(*np.shape(T))
            noise_flag ^= True
            if feature_class == 0 or feature_class == 1:
                feature_class = 4
            else:
                feature_class = 5

        # Join and normalize event contexts
        feedforward_event_context = T.reshape(input_dim, 1)
        feedforward_event_context /= np.linalg.norm(feedforward_event_context)
        recurrent_event_context = np.exp(rec_T-t) / recurrent_tau
        recurrent_event_context /= np.linalg.norm(recurrent_event_context)

        dist = (alpha*np.dot(w_ff, feedforward_event_context)  # cos(theta) = A*B/||A||/||B||
                + beta*np.dot(w_rec, recurrent_event_context))
        dist[dist < thres] = 0
        winner = np.argmax(dist)  # Looking for when theta=0 => cos(theta)=1
        if dist[winner] == 0:  # When no one reaches threshold
            thres = thres - thres_open

            # Keep track of winners and states
            winners_t.append(-1)
            feature_t.append(feature_class)

            if noise_flag:
                noise_flag ^= True
        else:
            w_ff[winner, :] = (1-eta_ff)*w_ff[winner, :] + eta_ff*feedforward_event_context.T
            w_ff[winner, :] = w_ff[winner, :]/np.linalg.norm(w_ff[winner, :])

            w_rec[winner, :] = (1-eta_rec)*w_rec[winner, :] + eta_rec*recurrent_event_context.T
            w_rec[winner, :] = w_rec[winner, :]/np.linalg.norm(w_rec[winner, :])

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
        if iteration % 500 == 0:
            w_mon.update_plots(w_ff, w_rec, thres.flatten(), True)

    monitor_class_winners.append(class_winners)
    class_winners = {0: [0 for _ in range(num_neurons)],
                     1: [0 for _ in range(num_neurons)],
                     2: [0 for _ in range(num_neurons)],
                     3: [0 for _ in range(num_neurons)],
                     4: [0 for _ in range(num_neurons)],
                     5: [0 for _ in range(num_neurons)]}

w_mon.update_plots(w_ff, w_rec, thres.flatten(), True)

plt.figure()
_ = plt.hist(noise_responses)
plt.xlabel('Neuron index')
plt.ylabel('# wins when noise was presented')

avg_hist = average_histograms(monitor_class_winners, num_neurons)
#plot_classifications(num_neurons, avg_hist, feature_mappings, color_mappings)
plot_classifications(num_neurons, monitor_class_winners[-1], feature_mappings, color_mappings)

plt.figure()
ax = plt.subplot(211)
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
for i in range(num_neurons):
    plt.plot(monitor_thres[i])
plt.xlabel('Time (samples)')
plt.ylabel('Thresholds')

plt.pause(0.1)
