import numpy as np

def generate_events(input_patterns):
    """ This function generates events from a input matrix.

    Parameters
    ----------
    input_patterns : numpy.ndarray
        Matrix with input pattern.
    samples : int
        Number of samples used in the simulation.

    Returns
    -------
    events_array : list of tuple
        Events with tuple (x, y, t) representing x-axis, y-axis, and time, respectivaly.
        
    """
    events_array = []
    num_samples = np.shape(input_patterns)[0]
    for sample in range(num_samples):
        for x, y in zip(*np.where(input_patterns[sample, :, :]==1)):
            events_array.append((x, y, sample))

    return events_array

def generate_observations(observations, num_samples,
                          feature_transition='random'):
    """ Generate observations from features.

    Parameters
    ----------
    observations : numpy.ndarray
        Basic features that will be used to generate observations.
    num_samples : int
        Number of samples to be generated

    Returns
    -------
    input_sequence : numpy.ndarray
    """
    x_dim, y_dim = np.shape(observations)[1], np.shape(observations)[2]
    input_sequence = np.zeros((num_samples, x_dim, y_dim))

    if feature_transition == 'random':
        feature_transition = np.random.choice
    elif feature_transition == 'random_walk':
        pass
    num_features = np.shape(observations)[0]
    for sample in range(num_samples):
        feature = feature_transition(range(num_features))
        input_sequence[sample, :, :] = observations[feature, :, :]

    return input_sequence
