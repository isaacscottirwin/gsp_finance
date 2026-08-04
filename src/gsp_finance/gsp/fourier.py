import numpy as np


def graph_fourier_transform(signal: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Compute the graph Fourier transform of a signal given the graph Laplacian.

    Args:
        signal (numpy.ndarray): The input signal defined on the graph nodes.
        eigenvectors (numpy.ndarray): The eigenvectors of the graph Laplacian.

    Returns:
        numpy.ndarray: The graph Fourier transform of the input signal.
    """
    fourier_transform = eigenvectors.T @ signal
    return fourier_transform

def inverse_graph_fourier_transform(coefficients: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Compute the inverse graph Fourier transform of the coefficients.

    Args:
        coefficients (numpy.ndarray): The graph Fourier coefficients.
        eigenvectors (numpy.ndarray): The eigenvectors of the graph Laplacian.

    Returns:
        numpy.ndarray: The reconstructed signal on the graph nodes.
    """
    signal = eigenvectors @ coefficients
    return signal

def dirichlet_energy(signal: np.ndarray, laplacian: np.ndarray) -> float:
    """Calculate the Dirichlet energy of a signal on the graph.

    Args:
        signal (numpy.ndarray): The input signal defined on the graph nodes.
        laplacian (numpy.ndarray): The Laplacian matrix of the graph.

    Returns:
        float: The Dirichlet energy of the signal.
    """
    energy = signal.T @ laplacian @ signal
    return energy
