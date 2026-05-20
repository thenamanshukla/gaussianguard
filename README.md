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
4. **Predictive Variance ($\Sigma_*$):** $\Sigma_* = K(X_*, X_*) - K_*^\top [K + \sigma_n^2 I]^{-1} K_*$
*Logic:* Calculates the remaining uncertainty. Near an anchor, the subtracted term increases, causing the variance (and the gray corridor) to shrink.

### Implementation Details
* **Matrix Dimensionality:** We use `X.reshape(-1, 1)` to transform flat lists into 2D column matrices. This is mandatory for Matrix multiplication and Dot Product operations in the Linear Algebra backend.
* **Resolution (60 vs 200):** We provide 60 "Anchor" points for training, but request **200 predictions** ($X_*$) to generate a high-resolution, continuous manifold for visualization.
* **3-Sigma Manifold:** The safety corridor covers **99.7%** of the probability mass ($\mu_* \pm 3\sigma_*$). Any observation breaching this fence is classified as a statistical anomaly.

 ## Day 1: Core Concept Intuition

### 1. From Bell Curves to Random Functions
A standard Gaussian (Bell Curve) defines a probability distribution over a **single variable** (scalar value) using a Mean ($\mu$) and Variance ($\sigma^2$). A **Gaussian Process (GP)** scales this concept to infinity: it defines a probability distribution over **entire continuous functions**. Instead of asking *"Which number is most likely?"*, a GP asks *"Which curve shape is most likely?"*

### 2. The Mechanics of Inference (The 3-Step Filter)
1. **The Infinite Prior:** We initialize a blank graph representing a vector space of infinite possible functions (flat lines, steep drops, random wiggles).
2. **The Kernel Scoring Engine:** We apply a **Kernel Function** to score these infinite curves. Smooth, structurally sound curves receive a high probability score; chaotic, discontinuous curves receive a score approaching zero. This is our **Prior Distribution**.
3. **The Data Constraint (Conditioning):** When observed real-world data points arrive, the GP acts as a filter. It instantly discards any function that does not pass through those observed coordinates. The remaining collection of surviving functions forms our **Posterior Distribution**.

### 3. The Kernel as a Structural Design Choice
The Kernel is our mathematical assumption about the *style* and *behavior* of the data stream. By selecting a kernel, we dictate the geometric properties (smoothness, periodicity, ruggedness) that the **KernelSentry** will consider "normal." 

>  **Theoretical Reference:** The baseline visual and conceptual framework for this architecture was inspired by the interactive visualization paradigm found in [*A Visual Exploration of Gaussian Processes* (Distill.pub)](https://distill.pub/2019/visual-exploration-gaussian-processes/). THIS IS DAY 1
