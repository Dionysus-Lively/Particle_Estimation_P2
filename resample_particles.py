# Function for resampling 1000 particles based on previous weights and locations

import numpy as np

def resample_particles(particles, norm_weights):
    """
    Resample particles based on their weights with some randomness to avoid particle depletion.

    Parameters:
    particles (2d list): A list of particle locations and headings, where each particle is represented as [x; y; heading].
    weights (list): A list of normalized weights corresponding to each particle-heading pair.

    Returns:
    list: A new list of resampled particles and headings.
    """

    location_range = 0.5 # Define the range for random noise injection
    heading_range = np.pi / 4 # Define the range for random noise injection

    # Initialize the random number generator
    rng = np.random.default_rng()

    # num_rand_particles

    # Resample particles based on normalized weights
    resampled_indices = np.random.choice(len(particles), size=len(particles), p=norm_weights)
    resampled_particles = [particles[:,i] for i in resampled_indices]

    # Define noise ranges
    pos_range = 0.5*100 # convert to distance beteween particles as represented by the map
    head_range = np.pi / 4 # 45 degree range in radians

    # Injecting noise
    for i in range(len(resampled_particles)):
        resampled_particles[i][0] += rng.integers(-pos_range, pos_range) # Inject noise to x-coordinate
        resampled_particles[i][1] += rng.integers(-pos_range, pos_range) # Inject noise to y-coordinate
        resampled_particles[i][2] += rng.integers(-head_range, head_range) # Inject noise to heading

    return resampled_particles