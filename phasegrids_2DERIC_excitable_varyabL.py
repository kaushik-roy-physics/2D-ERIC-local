# Code to generate the grids of phasemaps at a certain time for the 2D ERIC model + excitability, but for 
# varying parameters 'K, b, \Lambda'. Here we vary all three parameters and generate the snapshots at 
# a different 'K' value and fixed time. Each page of the pdf has phasemaps for varying 'b' and '\Lambda' 
# and fixed 'K' value. Different pages correspond to different 'K' values. We choose all of these parameters
# such that they put the theory in the weak coupling regime.
#
#
# The code can generate 10x10 and 5x5 grids. Comment/Uncomment the portions accordingly.
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

# Define the omega distribution

# Uniform distribution

def omega_uniform(w_min, w_max, N):         # N is the number of oscillators along each axis
    omega = np.random.uniform(w_min,w_max, size=(N,N))
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
    phase = np.random.uniform(-np.pi/2, np.pi/2 ,(N,N))  # narrowp3
    return phase

# completely random initial phases

def phase_random(N):
    phase = np.random.uniform(-np.pi, np.pi, (N, N))     # randomp
    return phase


# =============== Model Functions ===============

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

# function that computes the phases at the end of the time integration

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
    return X[-1, :, :]

# =============== Plotting and Utility Functions ===============

# Wrap the phase map function to allow passing a single tuple of parameters

def generate_plot_data_2DERIC_excitable(args):
    N, T, dt, K, L, phi_initial, omega_initial, b = args
    return phase_map_2DERIC_excitable(N, T, dt, K, L, phi_initial, omega_initial, b)

        
# Define the model parameters

w_min = 2*np.pi/180
w_max = 2*np.pi/150

# for use with truncated normal distributions

#mean = 2*np.pi/170
#scale = 2*np.pi*(1/160 - 1/170)

# for use with homogenous initial phases

#in_phase = np.pi/4

N = 50                            # number of oscillators: N^2
dt = 0.01                         # timestep

T = 1000                          # total integration time

# uncomment below for different times

#T = 1500
#T = 2000

# Define a list of desired b and \Lambda values

b_values = np.linspace(0.0, 0.04, 10)
L_values  = np.linspace(1.5, 2.0, 10)

# uncomment below for the 5x5 grids

#b_values = np.linspace(0.0, 0.04, 5)  
#L_values  = np.linspace(1.5, 2.0, 5) 

# Define a list of desired a values

a_values = [0.5, 0.6, 0.7, 0.8, 1.0]

# uncomment below for different ranges

#a_values = [0.5, 0.8, 1.0, 1.2, 1.5]
#a_values = [0.55, 0.65, 0.75, 0.85, 0.95]

# define the initial frequency and phase distributions for the computation

omega_initial = omega_uniform(w_min, w_max, N)      # uniform distribution

# uncomment below if using truncated normal distribution

#omega_initial = omega_trunc_normal(w_min, w_max, mean, scale, N)


phi_initial = phase_narrow(N)             # change the limits in original function for different results

# uncomment below for widest possible initial phase distribution

# phi_initial = phase_random(N)

start_time = time.time()

# Create a PDF file for the plots at the current T value

# customize output filename as necessary

pdf_filename = "phasemaps_narrowp3_uniformf_T1000_changeabL_10x10.pdf"
pdf_pages = PdfPages(pdf_filename)


for a in a_values:
    # Prepare a list of parameters for each iteration
    params = [(N, T, dt, a*(w_max - w_min) , L, phi_initial, omega_initial, b) for L in L_values for b in b_values]

    # Use multiprocessing to generate plot data
    with Pool() as p:
        plot_data = p.map(generate_plot_data_2DERIC_excitable, params)

    plt.figure(figsize=(30, 30))  # Adjust size as per your preference
    
# uncomment below for the 5x5 grids

#    plt.figure(figsize=(20, 20))

    for i, data in enumerate(plot_data):
        plt.subplot(10, 10, i + 1)

# uncomment below for the 5x5 grids

#        plt.subplot(5, 5, i + 1)

        plt.imshow(data, cmap='plasma')
        L, b = L_values[i // 10], b_values[i % 10]
        
# uncomment below for the 5x5 grids

#        L, b = L_values[i // 5], b_values[i % 5]

        plt.title(fr'b={b:.3f}, $\Lambda$={L:.2f}', fontsize = 18)
        plt.axis('off')  # Turn off axis numbers and ticks for clarity

        plt.tight_layout()

    pdf_pages.savefig()
    plt.clf()
pdf_pages.close()

print(f"Phase maps saved to {pdf_filename}")
    
end_time = time.time()

print("CPU computation took", end_time - start_time, "seconds")
