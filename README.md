#  Project Sentry: Gaussian Process Anomaly Detection

## Day 2 Milestone: Engine Calibration & Matrix Conditioning
On Day 2, we transitioned from raw signal generation to constructing a Bayesian "Sentry." This system learns the underlying manifold of a noisy data stream to establish a dynamic safety corridor.

###  The Logic: Posterior Conditioning
The Sentry does not utilize a hardcoded sine wave formula. Instead, it employs **Bayesian Inference**:
1. **The Prior:** Before observing data, the Sentry assumes a state of global uncertainty (Infinite Prior).
2. **The Constraints:** We provided **60 Discrete Observations** (Anchors) sampled from a noisy sine process.
3. **The Posterior:** The Sentry performs "Conditioning," crushing the uncertainty at these 60 points to reconstruct the most probable latent path.

###  The Mathematics

#### 1. The Kernel (The Smoothness Rule)
The **Radial Basis Function (RBF)** defines the covariance between time points:
$$k(x, x') = \sigma_f^2 \exp\left( -\frac{\|x - x'\|^2}{2l^2} \right)$$
* **$\sigma_f^2$ (Signal Variance):** Optimized to **~1.66**, defining the vertical probability boundary.
* **$l$ (Length-scale):** Optimized to **~2.36**, defining the horizontal "memory" of the signal.

#### 2. The Predictive Trace: Calculating $u_*$
To predict the signal value at a new time $X_*$ (e.g., Time = 1.5):
1. **Similarity Vector ($K_*$):** Measures the distance between $X_*$ and all 60 training dots.
2. **Weight Vector ($w$):** Calculated during training as $w = [K + \sigma_n^2 I]^{-1} y$.
3. **Predictive Mean:** The final height $\mu_*$ is the dot product $K_*^\top \cdot w$.
   - *Logic:* The Sentry calculates a weighted average where training points closer to $X_*$ exert a stronger pull on the prediction.

### Implementation Details
* **Matrix Dimensionality:** We use `X.reshape(-1, 1)` to transform flat lists into 2D column matrices. This is mandatory for Matrix multiplication and Dot Product operations in the Linear Algebra backend.
* **Resolution (60 vs 200):** We provide 60 "Anchor" points for training, but request **200 predictions** ($X_*$) to generate a high-resolution, continuous manifold for visualization.
* **3-Sigma Manifold:** The safety corridor covers **99.7%** of the probability mass ($\mu_* \pm 3\sigma_*$). Any observation breaching this fence is classified as a statistical anomaly.

