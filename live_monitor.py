import psutil
import numpy as np
import time
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

warnings.filterwarnings("ignore")

print("Sentry: Bayesian KernelGuard v3.0 (Self-Learning Mode)")
print("Phase 1: Observation Mode (60 seconds)...")

# 1. Initialize empty history
history_x = np.atleast_2d(np.linspace(0, 59, 60)).T
history_y = []

# 2. Observation Phase
while len(history_y) < 60:
    current_usage = psutil.cpu_percent(interval=1)
    history_y.append(current_usage)
    print(f"Learning Baseline... [{len(history_y)}/60]", end="\r")

print("\n Baseline Established. Transitioning to Active Guard.")
history_y = np.array(history_y)

# 3. Dynamic Kernel Setup 
# We use the average of your learning phase as the starting constant
initial_baseline = np.mean(history_y)
kernel = ConstantKernel(constant_value=initial_baseline) + 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel)

print(f"KernelGuard Active. Baseline: {initial_baseline:.1f}%")
print("-" * 60)

# 4. Active Guard Phase
try:
    while True:
        current_usage = psutil.cpu_percent(interval=1)
        
        # Update Window
        history_y = np.roll(history_y, -1)
        history_y[-1] = current_usage
        
        # Re-fit the brain
        gp.fit(history_x, history_y)
        
        # Predict
        mu, sigma = gp.predict(np.atleast_2d([60]).T, return_std=True)
        
        # 3-Sigma logic
        upper = mu[0] + 3 * sigma[0]
        lower = mu[0] - 3 * sigma[0]
        
        status = "NORMAL"
        if current_usage > upper or current_usage < lower:
            status = " ANOMALY!"
            
        print(f"CPU: {current_usage:5.1f}% | Sentry Expected: {mu[0]:5.1f} | Status: {status}")

except KeyboardInterrupt:
    print("\n[!] Sentry standing down.")
