# APBM CKF

The Augmented Physics-Based Model (APBM) is implemented with the Cubature Kalman Filter (CKF) in this project.

- The `tmlp` file contains the class definition for the Multilayer Perceptron (MLP) used in the learning process.

## Tracking Results

### Lorenz Attractor Model
<img src="figs/lorenz_seed0_adjusted.png" alt="Lorenz Attractor Tracking Result" width="600"/>

### Constant-Turning-Rate Model
<img src="figs/constant_velocity_seed9.png" alt="Constant-Turning-Rate Tracking Result" width="600"/>

 - For more details, refer to the paper:  
   ```
   Imbiriba, T., Demirkaya, A., Duník, J., Straka, O., Erdoğmuş, D. and Closas, P., 
   2022, July. Hybrid neural network augmented physics-based models for nonlinear filtering. In 2022 25th International Conference on Information Fusion (FUSION) (pp. 1-6). IEEE.
   ```
   [IEEE FUSION 2022: https://ieeexplore.ieee.org/document/9841291](https://ieeexplore.ieee.org/document/9841291)
