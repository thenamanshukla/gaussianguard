import sys
import os
import numpy as np

# Force Matplotlib to use a headless image backend so it never crashes on Linux setups
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bridge the pathing matrix so python can locate core/ and utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.signal_gen import generate_signal
from core.sentry_engine import KernelSentry

def main():
    print("Initializing Day 2 Sandbox Environment...")
    
    # 1. Harvest baseline sensor data
    X_train, y_train = generate_signal(n_samples=50, noise_level=0.15)
    
    # 2. Spin up the Sentry and feed it the baseline data
    sentry = KernelSentry()
    sentry.learn(X_train, y_train)
    
    # 3. Create dense future monitoring markers
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_mean, y_std = sentry.check_integrity(X_test)
    
    # 4. Generate the workspace plot asset
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, color='black', alpha=0.6, label='Observed Baseline Activity')
    plt.plot(X_test, y_mean, color='blue', label='Sentry Expectation (Mean μ)')
    plt.fill_between(X_test.ravel(), y_mean - 3*y_std, y_mean + 3*y_std, color='gray', alpha=0.2, label='3-Sigma Safety Corridor')
    
    plt.title("Day 2 Matrix Check: Sentry Verification Space")
    plt.xlabel("Time Horizon")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save chart file straight to the root folder
    plt.savefig('sentry_conditioned_space.png')
    print("Success! Execution matrix trace saved to 'sentry_conditioned_space.png'")

if __name__ == "__main__":
    main()
