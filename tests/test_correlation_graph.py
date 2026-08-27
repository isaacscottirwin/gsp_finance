import numpy as np
import pandas as pd
import pytest

from gsp_finance.scripts.build_graphs import BuildGraphs


class TestCorrelationGraph:
    """Tests for correlation-based market graph construction."""

    def setup_method(self) -> None:
        """Create a deterministic graph builder before each test."""
        base = np.array(
            [-3, -2, -1, 1, 2, 3],
            dtype=float,
        )

        returns = pd.DataFrame(
            {
                "A": base,
                "B": 2 * base,
                "C": -base,
                "D": [1, -1, 1, -1, 1, -1],
            },
            index=pd.date_range(
                "2025-01-01",
                periods=len(base),
            ),
        )

        # Bypass __init__ because it downloads the external dataset.
        self.graph_builder = BuildGraphs.__new__(BuildGraphs)

        self.graph_builder.returns = returns
        self.graph_builder.symbols = returns.columns.to_numpy()

        self.graph_builder.metadata = pd.DataFrame(
            {
                "Sector": {
                    "A": "Technology",
                    "B": "Technology",
                    "C": "Energy",
                    "D": "Energy",
                }
            }
        )

        self.symbol_to_index = {
            symbol: index
            for index, symbol in enumerate(
                self.graph_builder.symbols
            )
        }

    def test_graph_is_symmetric(self) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph()
        )

        dense = adjacency.toarray()

        np.testing.assert_allclose(
            dense,
            dense.T,
        )

    def test_graph_has_zero_diagonal(self) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph()
        )

        diagonal = adjacency.diagonal()

        np.testing.assert_allclose(
            diagonal,
            np.zeros_like(diagonal),
        )

    def test_positive_only_removes_negative_correlations(
        self,
    ) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph(
                absolute=False,
                positive_only=True,
            )
        )

        dense = adjacency.toarray()

        a = self.symbol_to_index["A"]
        b = self.symbol_to_index["B"]
        c = self.symbol_to_index["C"]

        assert dense[a, b] == pytest.approx(1.0)
        assert dense[a, c] == pytest.approx(0.0)

    def test_absolute_graph_preserves_negative_magnitude(
        self,
    ) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph(
                absolute=True,
                positive_only=False,
            )
        )

        dense = adjacency.toarray()

        a = self.symbol_to_index["A"]
        c = self.symbol_to_index["C"]

        assert dense[a, c] == pytest.approx(1.0)

    def test_graph_has_nonnegative_weights(self) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph(
                positive_only=True,
            )
        )

        assert np.all(adjacency.data >= 0.0)

    def test_threshold_graph_removes_weak_edges(
        self,
    ) -> None:
        threshold = 0.95

        adjacency = (
            self.graph_builder.build_threshold_graph(
                threshold=threshold,
                positive_only=True,
            )
        )

        assert np.all(
            adjacency.data >= threshold
        )

    def test_knn_graph_is_symmetric(self) -> None:
        adjacency = (
            self.graph_builder.build_knn_graph(
                k=1,
                absolute=True,
                mode="union",
            )
        )

        dense = adjacency.toarray()

        np.testing.assert_allclose(
            dense,
            dense.T,
        )

    def test_knn_graph_has_no_self_edges(self) -> None:
        adjacency = (
            self.graph_builder.build_knn_graph(
                k=1,
                absolute=True,
                mode="union",
            )
        )

        np.testing.assert_allclose(
            adjacency.diagonal(),
            0.0,
        )

    def test_laplacian_is_symmetric(self) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph()
        )

        laplacian = (
            self.graph_builder
            .combinatorial_laplacian(adjacency)
            .toarray()
        )

        np.testing.assert_allclose(
            laplacian,
            laplacian.T,
        )

    def test_laplacian_rows_sum_to_zero(self) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph()
        )

        laplacian = (
            self.graph_builder
            .combinatorial_laplacian(adjacency)
            .toarray()
        )

        row_sums = laplacian.sum(axis=1)

        np.testing.assert_allclose(
            row_sums,
            np.zeros(laplacian.shape[0]),
            atol=1e-12,
        )

    def test_laplacian_is_positive_semidefinite(
        self,
    ) -> None:
        adjacency = (
            self.graph_builder.build_correlation_graph()
        )

        laplacian = (
            self.graph_builder
            .combinatorial_laplacian(adjacency)
            .toarray()
        )

        eigenvalues = np.linalg.eigvalsh(laplacian)

        assert eigenvalues.min() >= -1e-10

    def test_sector_graph_connects_same_sector(
        self,
    ) -> None:
        adjacency = (
            self.graph_builder.build_sector_graph()
        )

        dense = adjacency.toarray()

        a = self.symbol_to_index["A"]
        b = self.symbol_to_index["B"]
        c = self.symbol_to_index["C"]

        assert dense[a, b] == pytest.approx(1.0)
        assert dense[a, c] == pytest.approx(0.0)