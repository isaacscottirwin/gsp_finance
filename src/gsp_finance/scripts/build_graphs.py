import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.csgraph import connected_components

from .download_data import SP500Data


class BuildGraphs:
    """
    Construct sparse stock-market graphs from S&P 500 return data.

    Each node represents a stock in a cleaned S&P 500 universe. Graph edges
    encode relationships between stocks, typically derived from historical
    return correlations.

    The class stores all adjacency matrices as SciPy CSR sparse matrices to
    reduce memory usage and support efficient sparse linear algebra.

    Parameters
    ----------
    n : int
        Number of stocks to include in the cleaned universe. Stocks are
        selected by the rules defined in ``SP500Data.get_clean_universe``.

    Attributes
    ----------
    sp500_data : SP500Data
        Loaded S&P 500 dataset object.

    returns : pandas.DataFrame
        Log-return matrix with dates as rows and stock symbols as columns.

    prices : pandas.DataFrame
        Adjusted-price matrix aligned with ``returns``.

    metadata : pandas.DataFrame
        Company metadata indexed by stock symbol.

    symbols : numpy.ndarray
        Stock symbols in the exact node ordering used by all adjacency
        matrices constructed by this class.

    Notes
    -----
    Maintaining a consistent node ordering is critical. Row and column ``i``
    of every adjacency matrix correspond to ``symbols[i]``.
    """

    def __init__(self, n: int) -> None:
        self.sp500_data = SP500Data().load()

        self.returns, self.prices, self.metadata = (
            self.sp500_data.get_clean_universe(number_of_stocks=n)
        )

        self.symbols = self.returns.columns.to_numpy()

    def _correlation_similarity(
        self,
        absolute: bool = False,
        positive_only: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the pairwise return-correlation similarity matrix.

        Parameters
        ----------
        absolute : bool, default=False
            If True, use absolute correlation magnitudes,

                s_ij = |rho_ij|.

            Strong positive and negative correlations are then treated as equally
            strong relationships.

        positive_only : bool, default=True
            If True and ``absolute`` is False, negative correlations are replaced
            by zero,

                s_ij = max(rho_ij, 0).

            This preserves the standard nonnegative-weight interpretation of the
            combinatorial graph Laplacian.

        Returns
        -------
        pandas.DataFrame
            Symmetric stock-stock similarity matrix with zero diagonal.

        Notes
        -----
        The diagonal is set to zero because self-edges are excluded.
        """
        similarity = self.returns.corr()

        if absolute:
            similarity = similarity.abs()
        elif positive_only:
            similarity = similarity.clip(lower=0.0)

        values = similarity.to_numpy(copy=True)
        np.fill_diagonal(values, 0.0)

        return pd.DataFrame(
            values,
            index=similarity.index,
            columns=similarity.columns,
        )

    def _correlation_distance(self) -> pd.DataFrame:
        """
        Compute pairwise correlation distance between stocks.

        The distance is defined as

            d_ij = sqrt(2 * (1 - rho_ij)).

        Returns
        -------
        pandas.DataFrame
            Symmetric stock-stock distance matrix with zero diagonal.
        """
        correlation = self.returns.corr()

        values = correlation.to_numpy(
            dtype=float,
            copy=True,
        )

        values = np.clip(values, -1.0, 1.0)

        distance = np.sqrt(
            2.0 * (1.0 - values)
        )

        np.fill_diagonal(distance, 0.0)

        return pd.DataFrame(
            distance,
            index=correlation.index,
            columns=correlation.columns,
        )

    def build_correlation_graph(
        self,
        absolute: bool = False,
        positive_only: bool = True,
    ) -> sp.csr_matrix:
        """
        Construct the full weighted correlation graph.

        Each pair of stocks is connected with edge weight equal to its processed
        return correlation,

            A_ij = s_ij.

        Parameters
        ----------
        absolute : bool, default=False
            Use absolute correlations as edge weights.

        positive_only : bool, default=True
            Replace negative correlations with zero when ``absolute`` is False.

        Returns
        -------
        scipy.sparse.csr_matrix
            Weighted symmetric adjacency matrix.

        Notes
        -----
        Although the result is stored in CSR format, this graph will generally be
        nearly dense because most stock pairs have nonzero correlation. For graph
        signal processing, thresholded or k-nearest-neighbor graphs may provide
        stronger locality and greater sparsity.
        """
        similarity = self._correlation_similarity(
            absolute=absolute,
            positive_only=positive_only,
        )

        adjacency = sp.csr_matrix(
            similarity.to_numpy(dtype=float)
        )

        adjacency.eliminate_zeros()

        return adjacency

    def build_threshold_graph(
        self,
        threshold: float,
        absolute: bool = False,
        positive_only: bool = True,
    ) -> sp.csr_matrix:
        """
        Construct a weighted correlation-threshold graph.

        An edge between stocks i and j is retained when

            s_ij >= threshold,

        where s_ij is the processed correlation similarity.

        Parameters
        ----------
        threshold : float
            Minimum similarity required for an edge.

        absolute : bool, default=False
            Use absolute correlations.

        positive_only : bool, default=True
            Replace negative correlations with zero when ``absolute`` is False.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse symmetric weighted adjacency matrix.

        Notes
        -----
        The threshold controls graph sparsity. Large thresholds create sparse graphs
        but may disconnect the graph; small thresholds retain more weak relationships.
        """
        if threshold < 0:
            raise ValueError("threshold must be nonnegative.")

        similarity = self._correlation_similarity(
            absolute=absolute,
            positive_only=positive_only,
        )

        values = similarity.to_numpy(dtype=float, copy=True)
        
        values[values < threshold] = 0.0

        adjacency = sp.csr_matrix(values)

        adjacency.eliminate_zeros()

        return adjacency

    def build_knn_graph(
        self,
        k: int,
        absolute: bool = False,
        positive_only: bool = True,
        mode: str = "union",
    ) -> sp.csr_matrix:
        """
        Build a sparse weighted k-nearest-neighbor correlation graph.

        Each stock selects the k stocks with the greatest similarity.

        Parameters
        ----------
        k
            Number of nearest neighbors per stock.

        absolute
            If True, use absolute return correlations.

        positive_only
            If True, negative correlations are set to zero.
            Ignored when absolute=True.

        mode
            "union":
                Keep edge i-j when either i selects j or j selects i.

            "mutual":
                Keep edge i-j only when both i selects j and j selects i.

        Returns
        -------
        scipy.sparse.csr_matrix
            Symmetric weighted adjacency matrix.
        """
        n = len(self.symbols)

        if not 1 <= k < n:
            raise ValueError(
                f"k={k} must satisfy 1 <= k < {n}."
            )

        if mode not in {"union", "mutual"}:
            raise ValueError(
                "mode must be either 'union' or 'mutual'."
            )

        similarity = self._correlation_similarity(
            absolute=absolute,
            positive_only=positive_only,
        )

        values = similarity.to_numpy(dtype=float, copy=True)

        rows = []
        cols = []
        data = []

        for i in range(n):
            row = values[i]

            # Indices of the k largest similarities.
            neighbor_indices = np.argpartition(
                row,
                -k,
            )[-k:]

            for j in neighbor_indices:
                weight = row[j]

                if weight > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(weight)

        directed = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
        )

        if mode == "union":
            adjacency = directed.maximum(directed.T)

        else:
            adjacency = directed.minimum(directed.T)

        adjacency.setdiag(0)
        adjacency.eliminate_zeros()

        return adjacency

    def build_minimum_spanning_tree(self) -> sp.csr_matrix:
        """
        Construct a minimum spanning tree using correlation distance.

        Pairwise stock distance is defined by

            d_ij = sqrt(2 * (1 - rho_ij)).

        The resulting graph contains all stocks and exactly n - 1 edges when the
        underlying distance graph is connected.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse symmetric adjacency matrix whose stored edge values are
            correlation distances.

        Notes
        -----
        A minimum spanning tree preserves only the minimum-distance backbone
        required to connect all stocks. It is useful for visualizing broad market
        structure but discards many relationships and therefore may be too sparse
        for some graph signal processing tasks.
        """
        distance = self._correlation_distance()

        dense = distance.to_numpy(
            dtype=float,
            copy=True,
        )

        graph = sp.csr_matrix(dense)

        mst = sp.csgraph.minimum_spanning_tree(graph)

        mst = mst + mst.T
        mst.eliminate_zeros()

        return mst.tocsr()

    def build_sector_graph(self, sector_column: str = "Sector") -> sp.csr_matrix:
        """
        Construct an unweighted graph connecting stocks in the same sector.

        Stocks i and j are connected when their metadata sector labels match,

            A_ij = 1  if sector_i = sector_j,
            A_ij = 0  otherwise.

        Parameters
        ----------
        sector_column : str, default="Sector"
            Metadata column containing sector labels.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse symmetric binary adjacency matrix.

        Notes
        -----
        This graph is metadata-driven rather than return-driven. It provides a
        useful baseline for comparing economically defined structure with
        correlation-derived structure.
        """
        if sector_column not in self.metadata.columns:
            raise KeyError(
                f"{sector_column!r} is not present in metadata."
            )

        sectors = (
            self.metadata
            .reindex(self.symbols)[sector_column]
            .to_numpy()
        )

        rows = []
        cols = []

        n = len(self.symbols)

        for i in range(n):
            if pd.isna(sectors[i]):
                continue

            for j in range(i + 1, n):
                if sectors[i] == sectors[j]:
                    rows.extend((i, j))
                    cols.extend((j, i))

        data = np.ones(len(rows), dtype=float)

        adjacency = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
        )

        return adjacency

    @staticmethod
    def combinatorial_laplacian(
        adjacency: sp.csr_matrix,
    ) -> sp.csr_matrix:
        """
        Compute the combinatorial graph Laplacian.

        For weighted adjacency matrix A, the combinatorial Laplacian is

            L = D - A,

        where D is the diagonal weighted-degree matrix,

            D_ii = sum_j A_ij.

        Parameters
        ----------
        adjacency : scipy.sparse.csr_matrix
            Weighted adjacency matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse combinatorial Laplacian.
        """
        degrees = np.asarray(
            adjacency.sum(axis=1)
        ).ravel()

        degree_matrix = sp.diags(
            degrees,
            format="csr",
        )

        laplacian = degree_matrix - adjacency

        return laplacian.tocsr()

    def to_networkx(
        self,
        adjacency: sp.csr_matrix,
    ) -> nx.Graph:
        """
        Convert a sparse adjacency matrix to a NetworkX graph and attach
        stock symbols and available metadata to the nodes.
        """
        graph = nx.from_scipy_sparse_array(
            adjacency,
            create_using=nx.Graph,
            edge_attribute="weight",
        )

        symbol_mapping = {
            i: symbol
            for i, symbol in enumerate(self.symbols)
        }

        graph = nx.relabel_nodes(
            graph,
            symbol_mapping,
        )

        for symbol in graph.nodes:
            if symbol not in self.metadata.index:
                continue

            row = self.metadata.loc[symbol]

            for column, value in row.items():
                graph.nodes[symbol][column] = value

        return graph

    def graph_diagnostics(
        self,
        adjacency: sp.csr_matrix,
    ) -> dict[str, int | float]:
        """
        Return basic diagnostics for a sparse graph.
        """
        n = adjacency.shape[0]

        if n == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "average_degree": 0.0,
                "minimum_degree": 0,
                "maximum_degree": 0,
                "number_of_components": 0,
            }

        # Number of nonzero neighbors for each node.
        degrees = np.diff(adjacency.indptr)

        # Symmetric undirected adjacency stores each edge twice.
        number_of_edges = adjacency.nnz // 2

        number_of_components = connected_components(
            adjacency,
            directed=False,
            return_labels=False,
        )

        density = (
            2 * number_of_edges
            / (n * (n - 1))
            if n > 1
            else 0.0
        )

        return {
            "nodes": n,
            "edges": number_of_edges,
            "density": float(density),
            "average_degree": float(degrees.mean()),
            "minimum_degree": int(degrees.min()),
            "maximum_degree": int(degrees.max()),
            "number_of_components": int(number_of_components),
        }

    def visualize_graph(
        self,
        adjacency: sp.csr_matrix,
        title: str = "Stock Graph",
        figsize: tuple[int, int] = (10, 10),
        node_size: int = 100,
        font_size: int = 8,
        
    ) -> None:
        """
        Visualize a sparse stock graph using a NetworkX spring layout.

        Parameters
        ----------
        adjacency : scipy.sparse.csr_matrix
            Sparse weighted adjacency matrix.

        title : str, default="Stock Graph"
            Plot title.

        figsize : tuple[int, int], default=(10, 10)
            Matplotlib figure size.

        node_size : int, default=100
            Node marker size.

        font_size : int, default=8
            Node-label font size.

        Notes
        -----
        This method converts the sparse matrix to a NetworkX graph only for
        visualization. Sparse SciPy matrices should remain the primary representation
        for graph construction and numerical computation.
        """
        import matplotlib.pyplot as plt

        graph = self.to_networkx(adjacency)

        plt.figure(figsize=figsize)

        pos = nx.spring_layout(graph)

        nx.draw(
            graph,
            pos=pos,
            with_labels=True,
            node_size=node_size,
            font_size=font_size,
            font_weight="bold",
            edge_color="gray",
            alpha=0.7,
        )

        plt.title(title, fontsize=16)
        plt.axis("off")
        plt.show()
        


    