import numpy as np

from .fourier import graph_fourier_transform, inverse_graph_fourier_transform


def simple_low_pass_filter(signal: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """
    Retain only the k lowest graph-frequency components.
    Assumes the columns of U are ordered by increasing Laplacian eigenvalue.
    """
    _validate_spectral_inputs(signal, U)
    if not 1 <= k <= U.shape[1]:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {U.shape[1]}.")
    U_low = U[:, :k]
    return inverse_graph_fourier_transform(graph_fourier_transform(signal, U_low),U_low)


def simple_high_pass_filter(signal: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """
    Retain only the k highest graph-frequency components.
    Assumes the columns of U are ordered by increasing Laplacian eigenvalue.
    """
    _validate_spectral_inputs(signal, U)
    if not 1 <= k <= U.shape[1]:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {U.shape[1]}.")
    U_high = U[:, -k:]
    return inverse_graph_fourier_transform(graph_fourier_transform(signal, U_high), U_high)


def reconstruction_error(original_signal: np.ndarray, reconstructed_signal: np.ndarray) -> float:
    """
    Calculate the squared Euclidean reconstruction error.
    """
    if original_signal.shape != reconstructed_signal.shape:
        raise ValueError("Original and reconstructed signals must have the same shape.")
    return np.linalg.norm(original_signal - reconstructed_signal) ** 2


def optimal_pass_filter(signal: np.ndarray, U: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retain the k graph-frequency components with the largest absolute Fourier coefficients."""
    _validate_spectral_inputs(signal, U)
    if not 1 <= k <= U.shape[1]:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {U.shape[1]}.")

    signal_hat = graph_fourier_transform(signal, U)
    selected_indices = np.argsort(np.abs(signal_hat))[-k:]
    signal_hat_k = np.zeros_like(signal_hat)
    signal_hat_k[selected_indices] = signal_hat[selected_indices]

    reconstructed_signal = inverse_graph_fourier_transform(signal_hat_k, U)

    return reconstructed_signal, selected_indices, signal_hat


def mid_band_pass_filter(
    signal: np.ndarray, eigenvalues: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """
    Retain the k graph-frequency components whose eigenvalues
    are closest to the numerical midpoint of the spectrum.
    """
    _validate_spectral_inputs(signal, U, eigenvalues)
    if not 1 <= k <= U.shape[1]:
        raise ValueError(f"k={k} must satisfy 1 <= k <= {U.shape[1]}.")
    lambda_mid = (eigenvalues.min() + eigenvalues.max()) / 2
    distances = np.abs(eigenvalues - lambda_mid)
    indices = np.argsort(distances)[:k]
    U_band = U[:, indices]
    return inverse_graph_fourier_transform(
        graph_fourier_transform(signal, U_band),
        U_band
    )


def band_pass_filter(signal: np.ndarray, eigenvalues: np.ndarray, U: np.ndarray, lambda_low: float, lambda_high: float) -> np.ndarray:
    """
    Apply a band-pass filter to a graph signal, retaining only the graph-frequency components
    whose Laplacian eigenvalues lie between lambda_low and lambda_high.
    This is useful for isolating specific frequency bands in the signal.
    """
    _validate_spectral_inputs(signal, U, eigenvalues)
    if lambda_low > lambda_high:
        raise ValueError("lambda_low must be less than or equal to lambda_high.")
    indices = np.where((eigenvalues >= lambda_low) & (eigenvalues <= lambda_high))[0]
    if len(indices) == 0:
        raise ValueError("No graph frequencies fall inside the requested band.")
    U_band = U[:, indices]
    return inverse_graph_fourier_transform(
        graph_fourier_transform(signal, U_band),
        U_band
    )


def band_notch_filter(signal: np.ndarray, eigenvalues: np.ndarray, U: np.ndarray, lambda_low: float, lambda_high: float) -> np.ndarray:
    """
    Apply a band-notch filter to a graph signal, removing components whose Laplacian eigenvalues
    lie between lambda_low and lambda_high, while retaining all other components.
    This is useful for removing specific frequency bands from the signal.
    """
    _validate_spectral_inputs(signal, U, eigenvalues)
    if lambda_low > lambda_high:
        raise ValueError("lambda_low must be less than or equal to lambda_high.")

    indices = np.where((eigenvalues < lambda_low) | (eigenvalues > lambda_high))[0]
    if len(indices) == 0:
        raise ValueError("No graph frequencies fall outside the requested band.")

    U_notch = U[:, indices]
    return inverse_graph_fourier_transform(graph_fourier_transform(signal, U_notch), U_notch)


def heat_kernel_filter(signal: np.ndarray, eigenvalues: np.ndarray, U: np.ndarray, tau: float) -> np.ndarray:
    """
    Apply the heat kernel filter to a graph signal.

    The heat kernel filter attenuates high-frequency components of the signal
    based on the Laplacian eigenvalues and the diffusion parameter tau.
    """
    _validate_spectral_inputs(signal, U, eigenvalues)
    if tau < 0:
        raise ValueError("tau must be >= 0")
    x_hat = graph_fourier_transform(signal, U)
    heat_response = np.exp(-tau * eigenvalues)
    y_hat = heat_response * x_hat
    return inverse_graph_fourier_transform(y_hat, U)


def tikhonov_filter(signal: np.ndarray, eigenvalues: np.ndarray, U: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply the Tikhonov regularization filter to a graph signal.

    The Tikhonov filter attenuates high-frequency components of the signal
    based on the Laplacian eigenvalues and the regularization parameter alpha.
    """
    _validate_spectral_inputs(signal, U, eigenvalues)
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    x_hat = graph_fourier_transform(signal, U)
    tikhonov_response = (1 / (1 + alpha * eigenvalues))
    y_hat = tikhonov_response * x_hat
    return inverse_graph_fourier_transform(y_hat, U)


def polynomial_filter(signal: np.ndarray, L: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Apply a polynomial filter to a graph signal.

    The polynomial filter is defined as a linear combination of powers of the
    graph Laplacian matrix L, with the given coefficients.
    """
    if signal.shape[0] != L.shape[0]:
        raise ValueError(f"signal length {signal.shape[0]} does not match "
                         f"the number of graph nodes {L.shape[0]}.")
    if L.shape[1] != L.shape[0]:
        raise ValueError("L must be a square matrix.")
    if len(coefficients) == 0:
        raise ValueError("coefficients cannot be empty.")

    polynomial_signal = np.zeros_like(signal, dtype=float)
    current_term = signal.copy()
    for coefficient in coefficients:
        polynomial_signal += coefficient * current_term
        current_term = L @ current_term

    return polynomial_signal


def _validate_spectral_inputs(
    signal: np.ndarray,
    U: np.ndarray,
    eigenvalues: np.ndarray | None = None
) -> None:
    n = U.shape[1]
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if U.ndim != 2:
        raise ValueError("U must be a two-dimensional matrix.")
    if signal.shape[0] != U.shape[0]:
        raise ValueError(
            f"Signal length {signal.shape[0]} does not match "
            f"the number of nodes {U.shape[0]}."
        )
    if eigenvalues is not None:
        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must be one-dimensional.")
        if eigenvalues.shape[0] != n:
            raise ValueError("The number of eigenvalues must match the number of eigenvectors.")
