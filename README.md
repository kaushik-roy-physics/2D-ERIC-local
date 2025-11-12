# 2D ERIC model and beyond


# 2D ERIC (2DE) model:

The 2D ERIC model is defined by the following dynamical equations,

$$ \Bigg( \frac{\mathrm{d}\theta_{i,j} (t) }{\mathrm{d}t} \Bigg)_{2DE}  = \omega_{i,j} + K \Big( \sin(\theta_{i-1,j}(t) -\theta_{i,j}(t) ) + \sin(\theta_{i+1,j}(t) -\theta_{i,j}(t) ) + \sin(\theta_{i,j-1}(t) -\theta_{i,j}(t) ) + \sin(\theta_{i,j+1}(t) -\theta_{i,j}(t) ) $$

$$ + \Lambda \big ( \sin^2(\theta_{i-1,j}(t) -\theta_{i,j} (t)) + \sin^2(\theta_{i+1,j}(t) -\theta_{i,j}(t) ) + \sin^2(\theta_{i,j-1}(t) -\theta_{i,j} (t)) + \sin^2(\theta_{i,j+1}(t) -\theta_{i,j}(t) ) \big) \Big)  $$

Here $\theta_{i,j}(t)$ represents the phase of the oscillator at the $(i,j)$-th site in the square lattice. The salient feature of the coupling is that it is asymmetric and contains a genuine higher order harmonic. 

# 2D ERIC model with excitability (2DE + ex):

We introduce excitability into the above model by introducing a simple $b\, \sin \theta_{i,j}(t)$ term, such that in the absence of coupling, the individual oscillators can be in a quiescent state ($\omega_{i,j} < b$) or oscillatory state ($\omega_{i,j}>b$). The dynamical equations can be similarly written as,

$$ \Bigg( \frac{\mathrm{d}\theta_{i,j} (t) }{\mathrm{d}t} \Bigg)_{2DE+ex}  = \omega_{i,j} - b\, \sin \theta_{i,j}(t) + K \Big( \sin(\theta_{i-1,j}(t) -\theta_{i,j}(t) ) + \sin(\theta_{i+1,j}(t) -\theta_{i,j}(t) ) + \sin(\theta_{i,j-1}(t) -\theta_{i,j}(t) ) + \sin(\theta_{i,j+1}(t) -\theta_{i,j}(t) ) $$

$$ + \Lambda \big ( \sin^2(\theta_{i-1,j}(t) -\theta_{i,j} (t)) + \sin^2(\theta_{i+1,j}(t) -\theta_{i,j}(t) ) + \sin^2(\theta_{i,j-1}(t) -\theta_{i,j} (t)) + \sin^2(\theta_{i,j+1}(t) -\theta_{i,j}(t) ) \big) \Big)  $$

# 2D Kuramoto model:

The dynamical equations for the 2D Kuramoto model with nearest neighbor coupling are given by:

 $$  \Bigg( \frac{\mathrm{d}\theta_{i,j}(t)}{\mathrm{d}t} \Bigg)_{KM} = \omega_{i,j} + K \Big[  \sin\big(\theta_{i,j+1}(t) -\theta_{i,j}(t)\big) + \sin\big(\theta_{i,j-1}(t) -\theta_{i,j}(t)\big) $$
 
 $$ + \sin\big(\theta_{i+1,j}(t) -\theta_{i,j}(t)\big) + \sin\big(\theta_{i-1,j}(t) -\theta_{i,j}(t)\big) \Big] $$

 # 2D Rectified KUramoto (ReKU) model:

 This model is inspired from the work by Ho et. al. (https://doi.org/10.1073/pnas.2401604121). When the nearest neighbors are coupled to each other, the phase dynamics is governed by the following dynamical equations:

  $$ \Bigg( \frac{\mathrm{d}\theta_{i,j}(t)}{\mathrm{d}t}\Bigg)_{\text{ReKU}}  = \omega_{i,j} + K \Big[  h_{\text{ReKU}} \big(\theta_{i,j+1}(t) -\theta_{i,j}(t)\big) + h_{\text{ReKU}} \big(\theta_{i,j-1}(t) -\theta_{i,j}(t)\big) $$
  
  $$ + h_{\text{ReKU}}\big(\theta_{i+1,j}(t) -\theta_{i,j}(t)\big) + h_{\text{ReKU}}\big(\theta_{i-1,j}(t) -\theta_{i,j}(t)\big) \Big] $$

The coupling function is given by: $ h_{\text{ReKU}}(\theta) = \text{max}(\sin \theta, 0) $.

# 2D Kuramoto model for quadratic integrate and fire (QIF) neurons:

The phase dynamics is described by the following equations:

$$ \Bigg( \frac{\mathrm{d}\theta_{i,j}(t)}{\mathrm{d}t}\Bigg)_{QIF}^{KM}  = \omega_{i,j} + K \Big[  \sin\big(\theta_{i,j+1}(t) -\theta_{i,j}(t)\big) + \sin\big(\theta_{i,j-1}(t) -\theta_{i,j}(t)\big) + \sin\big(\theta_{i+1,j}(t) -\theta_{i,j}(t)\big) + \sin\big(\theta_{i-1,j}(t) -\theta_{i,j}(t)\big) $$

$$ + \Lambda \Big\{ \Big( 1 -  \cos \big(\theta_{i,j+1}(t) -\theta_{i,j}(t) \big) \Big) + \Big( 1 -  \cos \big(\theta_{i,j-1}(t) -\theta_{i,j}(t) \big) \Big) + \Big( 1 -  \cos \big(\theta_{i+1,j}(t) -\theta_{i,j}(t) \big) \Big) + \Big( 1 -  \cos \big(\theta_{i-1,j}(t) -\theta_{i,j}(t) \big) \Big) \Big\} \Big] $$

# Details of the repository

This repository contains the Python codes for generating the figures and movies presented in the paper titled "Foci, waves, excitability: self-organization of phase waves in a model of asymmetrically coupled embryonic oscillators" by Kaushik Roy and Paul Francois. Most of the figures can be generated using the code blocks in the "2DERIC+excitable.ipynb" JuPyteR notebook. To generate the movies ($10\times 10$ grids of time evolving phasemaps), we have leveraged batched processing and just-in-time (JIT) compilation with JAX for GPU-accelerated computation. Our implementation simultaneously simulates all parameter combinations (in the $(a,\Lambda)$ or $(b,\Lambda)$ space) as a single batched operation on the GPU (NVIDIA GeForce RTX 4070, 8 GB VRAM), where each integration step updates hundreds of grids (e.g., $100$ parameter combinations $\times$ 50 $\times$50 spatial grids = $250,000$ cells) in parallel. Combined with optimized timepoint capture, i.e running simulations once to maximum time rather than re-simulating for each timepoint, and minimal CPU-GPU data transfers, computation times are reduced to $\mathcal{O}(100)$ seconds even for large-scale parameter sweeps. This represents an almost $100$x speedup compared to traditional CPU-based \texttt{multiprocessing} approaches. The implementation also supports CPU backends for systems without GPU acceleration, maintaining the computational efficiency of batched operations.

1. "phasegrids_gpu_mp4.py": For each model discussed in the paper, this code generates movies of time evolving phasemaps as 10x10 grids where each frame shows the phasemaps in the $(K=a\Delta_{\omega},\Lambda)$ parameter space at a specific time. The model used and the initial phases, frequencies used are clearly shown. The models used include 2D ERIC model with an asymmetric, biharmonic coupling function which is the primary model that we are interested in. In addition, it contains the 2D ERIC model with excitability and other models that we have mentioned in the supplement such as the 2D Kuramoto model, 2D Rectified KUramoto (ReKU) model and the 2D Kuramoto model for Quadratic-Integrate-and-Fire (QIF) neurons. The user can incorporate any other phase model into this modular code and do similar dynamical studies.

2. "phasegrids_gpu_truncnorm.py":  Same as above but samples the natural frequencies of the oscillators from a truncated normal distribution with user defined mean and scale.

3. "phasegrids_gpu_latticesize.py": Same as the code described in 1 but studying the dependence on lattice sizes.

4. "phasegrids_gpu_abl.py": This is specifically for the 2D ERIC model with excitability (2DE + ex) described above. This code generates movies of time evolving phasemaps as 10x10 grids where each frame shows the instantaneous phasemaps in the $(b,\Lambda)$ parameter space at a given $K=a\Delta_{\omega}$. 
