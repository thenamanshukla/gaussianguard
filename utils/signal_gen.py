import numpy as np

def generate_signal(n_samples=60, noise_level=0.15):
    """
    Generates a clean baseline sine wave with added Gaussian random jitter.
    Returns:
        X: Timestamps shaped for 2D inputs (n_samples, 1)
        y: Measured signal outputs (n_samples,)
    """
    np.random.seed(42)  # Keeps data identical every time we run it
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    
    # Base pattern is a sine wave
    y_clean = np.sin(X).ravel()
    
    # Add random measurement noise
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_clean + noise
    
    return X, y
