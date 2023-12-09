# Code to generate the grids of phasemaps at different times for the 2D Kuramoto model for QIF neurons.
# The code can generate the 10x10 and 5x5 grids. Comment/Uncomment the portions accordingly.
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

# Truncated normal distribution

def omega_trunc_normal(w_min, w_max, mean, scale, N):    

    # Define the parameters of the truncated Gaussian distribution
    a = (w_min - mean) / scale  # Lower bound
    b = (w_max - mean) / scale  # Upper bound


    # Generate random samples from the truncated Gaussian distribution
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
    # Change the initial phase distribution width according to the need
    
    phase = np.random.uniform(-np.pi/2 , np.pi/2 , (N,N))     # narrowp3
    return phase

# completely random initial phases

def phase_random(N):
    phase = np.random.uniform(-np.pi, np.pi, (N, N))          # randomp
    return phase

# =============== Model Functions ===============

# 2D Kuramoto model for QIF neurons

# Please note that the L in the code below is actually 2 * \Lambda in the paper
# We have rewritten (1 - cos \theta) = 2 * sin^2 (\theta/2) for the code implementation

def kuramoto_QIF(phi, N, K, omega, L):        
    phi_up = np.roll(phi, -1, axis=0)            # getting nearest neighbors
    phi_up [-1, :]= phi[-1, :]                   # imposing open boundary conditions
    phi_down = np.roll(phi, 1, axis=0)
    phi_down[0, :] = phi[0, :]
    phi_left = np.roll(phi, -1, axis=1)
    phi_left[:, -1] = phi[:, -1]
    phi_right = np.roll(phi, 1, axis=1)
    phi_right[:, 0] = phi[:, 0]

    dphi_dt = np.zeros((N, N))
    dphi_dt += np.sin(phi_up-phi) + np.sin(phi_down-phi) + L*(np.sin( (phi_up-phi)/2 )**2) + L*(np.sin( (phi_down-phi)/2 )**2) 
    dphi_dt += np.sin(phi_left-phi) + np.sin(phi_right-phi) + L*(np.sin( (phi_left-phi)/2 )**2) + L*(np.sin( (phi_right-phi)/2 )**2) 
    dphi_dt = omega + K * dphi_dt                # computing the phase velocity
    
    return dphi_dt


# =============== Simulation Functions ===============

# function that computes the phases at the end of the time integration

def phase_map_QIF(N, T, dt, K, L, phi_initial, omega_initial):
    num_steps = int(T / dt) + 1
    X = np.zeros((num_steps, N, N))  # Storing the phases

    omega = omega_initial.copy()     # ensuring that the same initial
    phi = phi_initial.copy()         # conditions are used throughout
    
    # RK4 method of time integration
    
    for t in range(num_steps):
        k1 = kuramoto_QIF(phi, N, K, omega, L)
        k2 = kuramoto_QIF(phi + 0.5 * dt * k1, N, K, omega, L)
        k3 = kuramoto_QIF(phi + 0.5 * dt * k2, N, K, omega, L)
        k4 = kuramoto_QIF(phi + dt * k3, N, K, omega, L)

        phi += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi = np.angle(np.exp(1j * phi))  # Ensuring that phi is between -pi and pi
        X[t, :, :] = phi
    return X[-1, :, :]

# =============== Plotting and Utility Functions ===============

# Wrap the phase map function to allow passing a single tuple of parameters
def generate_plot_data_QIF(args):
    N, T, dt, K, L, phi_initial, omega_initial = args
    return phase_map_QIF(N, T, dt, K, L, phi_initial, omega_initial)

        
# Define the model parameters

w_min = 2*np.pi/180
w_max = 2*np.pi/150

# for use with truncated normal distributions

#mean = 2*np.pi/165
#scale = 2*np.pi*(1/160 - 1/165)

# for use with homogenous initial phases

#in_phase = np.pi/4


N = 50                                 # number of oscillators: N^2
dt = 0.01                              # time step

# adjust the ranges as necessary

a_values = np.linspace(0.5, 2.0, 10)   # range of a in K = a \Delta_{\omega}
L_values = np.linspace(2.4, 3.2, 10)   # range of \Lambda = [1.2 - 1.6]

# uncomment below for the 5x5 grids

#a_values = np.linspace(0.5, 2.0, 5)  # range of a in K = a \Delta_{\omega}
#L_values  = np.linspace(2.4, 3.2, 5) # range of \Lambda = [1.2 - 1.6]

# define the initial frequency and phase distributions for the computation

omega_initial = omega_uniform(w_min, w_max, N)                # uniform distribution

# uncomment below if using truncated normal distribution

#omega_initial = omega_trunc_normal(w_min, w_max, mean, scale, N)

phi_initial = phase_narrow(N)               # change the limits in original function for different results             
# uncomment below for widest possible initial phase distribution

#phi_initial = phase_random(N)

start_time = time.time()

# Define a list of desired T values

T_values = [1000, 1500, 2000]                # adjust as necessary

# Create a PDF file for the plots at the current T value

# uncomment to save pdf file

#pdf_filename = "phasemapsQIF_narrowp3_uniformf_QIF_10x10.pdf"
pdf_pages = PdfPages(pdf_filename)

for T in T_values:
    # Prepare a list of parameters for each iteration
    params = [(N, T, dt, a * (w_max - w_min) , L, phi_initial, omega_initial) for a in a_values for L in L_values]

    # Use multiprocessing to generate plot data
    with Pool() as p:
        plot_data = p.map(generate_plot_data_QIF, params)

    plt.figure(figsize=(30, 30))  # Adjust size as per your preference
    
# uncomment below for the 5x5 grids 

#    plt.figure(figsize=(20, 20)) 

    for i, data in enumerate(plot_data):
        plt.subplot(10, 10, i + 1)
        
# uncomment below for the 5x5 grids 

#        plt.subplot(5, 5, i + 1)

        plt.imshow(data, cmap='plasma')
        a, L = a_values[i // 10], L_values[i % 10]
        
# uncomment below for the 5x5 grids  

#        a, L = a_values[i // 5], L_values[i % 5]

        plt.title(fr'a={a:.2f}, $\Lambda$={L/2:.2f}', fontsize = 18)  # Note that here we have \Lambda = L/2
        plt.axis('off')  # Turn off axis numbers and ticks for clarity

        plt.tight_layout()

    pdf_pages.savefig()
    plt.clf()
pdf_pages.close()

print(f"Phase maps saved to {pdf_filename}")
    
end_time = time.time()

print("CPU computation took", end_time - start_time, "seconds")
