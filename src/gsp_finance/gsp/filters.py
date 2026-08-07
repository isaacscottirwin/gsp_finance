import numpy as np

from .fourier import graph_fourier_transform, inverse_graph_fourier_transform


def simple_low_pass_filter(signal: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """
    Retain only the k lowest graph-frequency components.
    """
    n = U.shape[1]
    if signal.shape[0] != U.shape[0]:
        raise ValueError(
            f"Signal length {signal.shape[0]} does not match "
            f"the number of graph nodes {U.shape[0]}.")
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}.")
    U_low = U[:, :k]
    return inverse_graph_fourier_transform(graph_fourier_transform(signal, U_low),U_low)


def simple_high_pass_filter(signal: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """
    Retain only the k highest graph-frequency components.
    """
    n = U.shape[1]
    if signal.shape[0] != U.shape[0]:
        raise ValueError(
            f"Signal length {signal.shape[0]} does not match "
            f"the number of graph nodes {U.shape[0]}.")
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}.")
    U_high = U[:, -k:]
    return inverse_graph_fourier_transform(graph_fourier_transform(signal, U_high), U_high)


def reconstruction_error(original_signal: np.ndarray, reconstructed_signal: np.ndarray) -> float:
    """
    Calculate the reconstruction error between the original and reconstructed signals.
    """
    return np.linalg.norm(original_signal - reconstructed_signal) ** 2

def optimal_pass_filter(signal: np.ndarray, U: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = U.shape[1]
    if signal.shape[0] != U.shape[0]:
            raise ValueError(f"the signal length: {signal.shape[0]} does not match the number of nodes: {U.shape[0]}. It needs to.")
    if k > n:
        raise ValueError(f"The value k: {k} must be less than of equal the number of eigenvectors: {n}")

    signal_hat = graph_fourier_transform(signal, U)
    selected_indicies = np.argsort(np.abs(signal_hat))[-k:]
    signal_hat_k = np.zeros_like(signal_hat)
    signal_hat_k[selected_indicies] = signal_hat[selected_indicies]

    reconstructed_signal = inverse_graph_fourier_transform(signal_hat_k, U)

    return reconstructed_signal, selected_indicies, signal_hat

def band_pass_filter(
    signal: np.ndarray, eigenvalues: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """
    Retain the k graph-frequency components whose eigenvalues
    are closest to the numerical midpoint of the spectrum.
    """
    n = U.shape[1]
    if signal.shape[0] != U.shape[0]:
        raise ValueError(
            f"Signal length {signal.shape[0]} does not match "
            f"the number of nodes {U.shape[0]}.")
    if len(eigenvalues) != n:
        raise ValueError("The number of eigenvalues must match the number of eigenvectors.")

    if not 1 <= k <= n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}.")
    lambda_mid = (eigenvalues.min() + eigenvalues.max()) / 2
    distances = np.abs(eigenvalues - lambda_mid)
    indices = np.argsort(distances)[:k]
    U_band = U[:, indices]
    return inverse_graph_fourier_transform(
        graph_fourier_transform(signal, U_band),
        U_band
    )


