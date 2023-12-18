# Code to generate the animations for the phasemaps for the 2D ERIC model + excitability.
# The 2D ERIC model can be obtained by putting b = 0 for the 2DE+ex model
#
# Author: Kaushik Roy
#
# Current affiliation: Dept. of Biosciences, Rice University, TX 77005, USA
# For correspondence, please email: kr70@rice.edu
#
#

# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.stats import truncnorm

# =============== Initialization Functions ===============

# Legacy seed for reproducibility

np.random.seed(12345)

# Define the omega distribution

# Uniform distribution

def omega_uniform(w_min, w_max, N):      # N is the number of oscillators along each axis
    omega = np.random.uniform(w_min, w_max, size=(N,N))
    return omega

# Truncated normal distribution

def omega_trunc_normal(w_min, w_max, mean, scale, N):    

    # Define the parameters of the truncated normal distribution
    a = (w_min - mean) / scale  # Lower bound
    b = (w_max - mean) / scale  # Upper bound


    # Generate random samples from the truncated Gaussian distribution
    omega = truncnorm.rvs(a, b , loc = mean , scale = scale , size = (N,N))
    
    return omega

# Homogenous initial phases

def phase_homogenous(N,angle):     
    phase = np.ones((N,N))*angle   # the angle variable can be set to any value between -pi and pi
    return phase

# Initial phase distributions of width less than 2*pi

# We use the following nomenclature for the distributions
# narrowp1: np.random.uniform(0, np.pi/2, (N,N))
# narrowp2: np.random.uniform(0, np.pi, (N,N))
# narrowp3: np.random.uniform(-np.pi/2, np.pi/2, (N,N))
# narrowp4: np.random.uniform(-np.pi/4, np.pi/4, (N,N))
# widerp1: np.random.uniform(-np.pi/8, np.pi, (N,N))
# widerp2: np.random.uniform(-np.pi/4, np.pi, (N,N))
# widerp3: np.random.uniform(-3*np.pi/8, np.pi, (N,N))
# widerp4: np.random.uniform(-np.pi/2, np.pi, (N,N))

def phase_narrow(N):
    phase = np.random.uniform(- np.pi/2, np.pi/2 ,(N,N))   # narrowp3
    return phase

# completely random initial phases

def phase_random(N):
    phase = np.random.uniform(-np.pi, np.pi, (N, N))       # randomp
    return phase

# =============== Model Functions ===============

# 2D ERIC model

def ERIC_2D(phi,N, K, omega, L):
    phi_up = np.roll(phi, -1, axis=0)         # getting nearest neighbors
    phi_up [-1, :]= phi[-1, :]                # imposing open boundary conditions
    phi_down = np.roll(phi, 1, axis=0)
    phi_down[0, :] = phi[0, :]
    phi_left = np.roll(phi, -1, axis=1)
    phi_left[:, -1] = phi[:, -1]
    phi_right = np.roll(phi, 1, axis=1)
    phi_right[:, 0] = phi[:, 0]

    dphi_dt = np.zeros((N, N))
    dphi_dt += np.sin(phi_up - phi) + np.sin(phi_down - phi) + L * (np.sin(phi_up - phi) ** 2) + L * (
                np.sin(phi_down - phi) ** 2)
    dphi_dt += np.sin(phi_left - phi) + np.sin(phi_right - phi) + L * (np.sin(phi_left - phi) ** 2) + L * (
                np.sin(phi_right - phi) ** 2)
    dphi_dt = omega + K * dphi_dt            # computing the phase velocity

    return dphi_dt


# 2D ERIC model + excitability

def ERIC_2D_excitable(phi, N, K, omega, L, b):        
    phi_up = np.roll(phi, -1, axis=0)         # getting nearest neighbors
    phi_up [-1, :]= phi[-1, :]                # imposing open boundary conditions
    phi_down = np.roll(phi, 1, axis=0)
    phi_down[0, :] = phi[0, :]
    phi_left = np.roll(phi, -1, axis=1)
    phi_left[:, -1] = phi[:, -1]
    phi_right = np.roll(phi, 1, axis=1)
    phi_right[:, 0] = phi[:, 0]

    dphi_dt = np.zeros((N, N))
    dphi_dt += np.sin(phi_up-phi) + np.sin(phi_down-phi) + L*(np.sin(phi_up-phi)**2) + L*(np.sin(phi_down-phi)**2) 
    dphi_dt += np.sin(phi_left-phi) + np.sin(phi_right-phi) + L*(np.sin(phi_left-phi)**2) + L*(np.sin(phi_right-phi)**2) 
    dphi_dt = omega - b * np.sin(phi) + K * dphi_dt      # computing the phase velocity
    
    return dphi_dt


# =============== Simulation Functions ===============

def animate_phases(X, dt, filename):
    fig, ax = plt.subplots()

    # Add title above the animation
    
    # Change as necessary
    
    fig.suptitle(r'$\Delta \theta_{in}=2\pi$, $K=0.8(\frac{1}{150}-\frac{1}{180})$ min$^{-1}$, $\Lambda=1.7$, $b = 0.0$', fontsize=14, fontweight='bold')

    # Initialize the image plot using 'twilight' colormap
    im = ax.imshow(X[0, :, :], cmap='twilight')

    # Set the colorbar to use the 'twilight' colormap and normalize the color range
    cbar = fig.colorbar(im, cmap='twilight', norm=plt.Normalize(-np.pi, np.pi))

    # Initialize the text annotation for time elapsed
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

    # Text below the animation
    
    # Change as necessary
    
    fig.text(0.5, 0.01, 'Spiral phase waves in 2DE', ha='center', va='bottom')

    num_steps = X.shape[0]

    num_frames = 15000  # Adjust the number of frames to shorten the video
    
    
    # function to update frame
    def update_frame(i):
        idx = int(num_steps / num_frames * i)
        im.set_array(X[idx, :, :])
        im.set_cmap('twilight')  # Ensure the colormap is set to 'twilight' during updates
        im.set_norm(plt.Normalize(-np.pi, np.pi))
        time_text.set_text('Time Elapsed: {:.1f}'.format(idx * dt))
        return im, time_text
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=30)

    # Adjust layout to prevent overlap of title and text
    fig.subplots_adjust(top=0.85, bottom=0.15)

    # Set up the writer
    Writer = animation.writers['ffmpeg']
    
    # adjust fps and num_frames to decide length of final movie
    
    writer = Writer(fps=500, bitrate=-1)   # length of final video = num_frames / fps
    
    
    # Save the animation as a video file
    anim.save(filename, writer=writer)


# function that computes the phases at the end of the time integration

def phase_map_2DERIC(N, T, dt, K, L, phi_initial, omega_initial):
    num_steps = int(T / dt)
    X = np.zeros((num_steps, N, N))  # Storing the cosine of the phases

    omega = omega_initial.copy()
    phi = phi_initial.copy()
    
    for t in range(num_steps):
        k1 = ERIC_2D(phi, N, K, omega, L)
        k2 = ERIC_2D(phi + 0.5 * dt * k1, N, K, omega, L)
        k3 = ERIC_2D(phi + 0.5 * dt * k2, N, K, omega, L)
        k4 = ERIC_2D(phi + dt * k3, N, K, omega, L)

        phi += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi = np.angle(np.exp(1j * phi))  # Ensuring that phi is between -pi and pi
        X[t, :, :] = phi
        
    animate_phases(X, dt, filename)

def phase_map_2DERIC_excitable(N, T, dt, K, L, phi_initial, omega_initial, b):
    num_steps = int(T / dt) + 1
    X = np.zeros((num_steps, N, N))  # Storing the phases

    omega = omega_initial.copy()     # ensuring that the same initial
    phi = phi_initial.copy()         # conditions are used throughout

    for t in range(num_steps):
        k1 = ERIC_2D_excitable(phi, N, K, omega, L, b)
        k2 = ERIC_2D_excitable(phi + 0.5 * dt * k1, N, K, omega, L, b)
        k3 = ERIC_2D_excitable(phi + 0.5 * dt * k2, N, K, omega, L, b)
        k4 = ERIC_2D_excitable(phi + dt * k3, N, K, omega, L, b)

        phi += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi = np.angle(np.exp(1j * phi))  # Ensuring that phi is between -pi and pi
        X[t, :, :] = phi

    animate_phases(X, dt, filename)

    

# Define the model parameters
N = 50
T = 2000
dt = 0.1

# We can use phase_map_2DERIC_excitable to obtain animations for the 2D ERIC case by setting b = 0.0

# b = 0.0 

# Excitability parameter for all oscillatory cells

#b = 0.01    

# Excitability parameter for all mixed population

#b = 2*np.pi/170

# Excitability parameter for all excitable cells

#b = 2*np.pi/145


w_min = 2*np.pi / 180
w_max = 2*np.pi / 150

K = 0.8 * (w_max - w_min)
L = 1.7

# Initial phase distributions narrower than 2*pi

#phi_initial = phase_narrow(N)

# Initial phase distribution of width 2*pi

phi_initial = phase_random(N)

# Using uniform distribution of natural frequencies

omega_initial = omega_uniform(w_min, w_max, N)

# Change filename as desired

filename = "phasemap_randomp_uniformf_2DERIC_final.mp4"

#filename = "phasemap_narrowp3_uniformf_alloscillatory_2DERIC_excitable.mp4"


# Call function to generate the animation

phase_map_2DERIC(N, T, dt, K, L, phi_initial, omega_initial)

#phase_map_2DERIC_excitable(N, T, dt, K, L, phi_initial, omega_initial, b)













