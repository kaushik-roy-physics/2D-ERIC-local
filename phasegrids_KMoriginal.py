# Code to generate the grids of phasemaps at different times for the 2D Kuramoto model.
# The code can generate a 5x5 grid. Make adjustments to generate different shape grids.
# We use multiprocessing module in Python to parallelize the computation.
# This code is particularly fast and efficient when run on a multiple core CPU server.
#
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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import truncnorm
from multiprocessing import Pool
import time

# =============== Initialization Functions ===============

# Legacy seed for reproducibility
np.random.seed(12345)

# Define the omega distributions

# Uniform distribution
def omega_uniform(w_min, w_max, N):         # N is the number of oscillators along each axis
    omega = np.random.uniform(w_min,w_max, size=(N,N))
    return omega

def omega_trunc_normal(w_min, w_max, mean, scale, N):    

    # Define the parameters of the truncated normal distribution
    a = (w_min - mean) / scale  # Lower bound
    b = (w_max - mean) / scale  # Upper bound


    # Generate random samples from the truncated original distribution
    omega = truncnorm.rvs(a, b , loc = mean , scale = scale , size = (N,N))
    
    return omega

# Homogenous initial phases
def phase_homogenous(N,angle):     # N is the number of oscillators along each axis
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
    phase = np.random.uniform(-np.pi/2 , np.pi/2 , (N,N)) # narrowp3
    return phase

# completely random initial phases
def phase_random(N):
    phase = np.random.uniform(-np.pi, np.pi, (N, N))      # randomp
    return phase

# =============== Model Functions ===============

# 2D Kuramoto model

def kuramoto_model(phi, N, K, omega):        
    phi_up = np.roll(phi, -1, axis=0)     # getting nearest neighbors
    phi_up [-1, :]= phi[-1, :]            # imposing open boundary conditions
    phi_down = np.roll(phi, 1, axis=0)
    phi_down[0, :] = phi[0, :]
    phi_left = np.roll(phi, -1, axis=1)
    phi_left[:, -1] = phi[:, -1]
    phi_right = np.roll(phi, 1, axis=1)
    phi_right[:, 0] = phi[:, 0]

    dphi_dt = np.zeros((N, N))
    dphi_dt += np.sin(phi_up-phi) + np.sin(phi_down-phi) 
    dphi_dt += np.sin(phi_left-phi) + np.sin(phi_right-phi)  
    dphi_dt = omega + K * dphi_dt         # computing the phase velocity
    
    return dphi_dt


# =============== Simulation Functions ===============

# function that computes the phases at the end of the time integration

def phase_maps_KMoriginal(N, T, dt, K, phi_initial, omega_initial):
    num_steps = int(T / dt) + 1
    X = np.zeros((num_steps, N, N))  # Storing the phases

    omega = omega_initial.copy()     # ensuring that the same initial
    phi = phi_initial.copy()         # conditions are used throughout
    
    # RK4 method of time integration
    for t in range(num_steps):
        k1 = kuramoto_model(phi, N, K, omega)
        k2 = kuramoto_model(phi + 0.5 * dt * k1, N, K, omega)
        k3 = kuramoto_model(phi + 0.5 * dt * k2, N, K, omega)
        k4 = kuramoto_model(phi + dt * k3, N, K, omega)

        phi += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi = np.angle(np.exp(1j * phi))  # Ensuring that phi is between -pi and pi
        X[t, :, :] = phi
    return X[-1, :, :]


# =============== Plotting and Utility Functions ===============

# Wrap the phase map function to allow passing a single tuple of parameters
def generate_plot_data_KMoriginal(args):
    N, T, dt, K, phi_initial, omega_initial = args
    return phase_maps_KMoriginal(N, T, dt, K, phi_initial, omega_initial)

# Define the model parameters
w_min = 2*np.pi/180
w_max = 2*np.pi/150

# for use with truncated normal distributions
#mean = 2*np.pi/165
#scale = 2*np.pi*(1/155 - 1/165)

# for use with homogenous initial phases
#in_phase = np.pi/4


N = 50                    # number of oscillators: N^2
dt = 0.01                 # time step


# adjust the range as necessary
a_values = np.linspace(0.5, 3.0, 25)       # range of a in K = a \Delta_{\omega}

# define the initial frequency and phase distributions for the computation

omega_initial = omega_uniform(w_min, w_max, N)       # uniform distribution

# uncomment below if using truncated normal distribution
#omega_initial = omega_trunc_normal(w_min, w_max, mean, scale, N)

phi_initial = phase_narrow(N)             # change the limits in original function for different results

# uncomment below for widest possible initial phase distribution
#phi_initial = phase_random(N)


start_time = time.time()

# Define a list of desired T values
T_values = [1000, 1500, 2000]             # adjust as necessary

# Create a PDF file for the plots at the current T value

# customize output filename if necessary

pdf_filename = "phasemaps_KMoriginal_narrowp3_uniformf_5x5_apoint5to3.pdf"
pdf_pages = PdfPages(pdf_filename)


for T in T_values:
    # Prepare a list of parameters for each iteration
    params = [(N, T, dt, a * (w_max - w_min) , phi_initial, omega_initial) for a in a_values]

    # Use multiprocessing to generate plot data
    with Pool() as p:
        plot_data = p.map(generate_plot_data_KMoriginal, params)

 # Prepare the plot for the current T value
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    for idx, data in enumerate(plot_data):
        ax = axes[idx // 5, idx % 5]  # Get the current axis
        ax.imshow(data, cmap="plasma", origin="lower", extent=[0, N, 0, N])
        ax.axis("off")
        ax.set_title(f'a={a_values[idx]:.2f}', fontsize=16)

    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close()

pdf_pages.close()


print(f"Phase maps saved to {pdf_filename}")
    
end_time = time.time()

print("CPU computation took", end_time - start_time, "seconds")
