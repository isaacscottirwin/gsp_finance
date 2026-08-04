import numpy as np


def simple_low_pass_filter(signal: np.ndarray, eigenvectors: np.ndarray, k: int ) -> np.ndarray:
    """
    A simple low-pass filter that retains only the first k Fourier coefficients of the signal.
    """
    if k > len(eigenvectors):
        raise ValueError(f"The value k: {k} must be less than of equal the number of eigenvectors: {len(eigenvectors)}")
    U_low = eigenvectors[:, :k]
    return U_low @ (U_low.T @ signal)
