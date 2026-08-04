import numpy as np


def combinatorial_laplacian(adj):
    """Calculate the combinatorial Laplacian matrix of a graph given its adjacency matrix.

    Args:
        adj (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
        numpy.ndarray: The Laplacian matrix of the graph.
    """
    degree_matrix = np.diag(np.sum(adj, axis=1))
    laplacian_matrix = degree_matrix - adj
    return laplacian_matrix

def normalized_laplacian(adj):
    """Calculate the normalized Laplacian matrix of a graph given its adjacency matrix.

    Args:
        adj (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
        numpy.ndarray: The normalized Laplacian matrix of the graph.
    """
    degree_matrix = np.diag(np.sum(adj, axis=1))
    inverse_sqrt_degree = 1 / np.sqrt(degree_matrix)
    inverse_sqrt_degree[np.isinf(inverse_sqrt_degree)] = 0  # Handle division by zero for isolated nodes
    normalized_laplacian_matrix = np.eye(adj.shape[0]) - inverse_sqrt_degree @ adj @ inverse_sqrt_degree
    return normalized_laplacian_matrix