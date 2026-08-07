from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd


class SP500Data:
    DATASET_NAME = "andrewmvd/sp-500-stocks"

    REQUIRED_FILES = {  # noqa: RUF012
        "sp500_stocks.csv",
        "sp500_companies.csv",
        "sp500_index.csv",
    }

    STOCK_NUMERIC_COLUMNS = [  # noqa: RUF012
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]

    def __init__(self) -> None:
        self.dataset_path: Path | None = None

        self.stocks: pd.DataFrame | None = None
        self.companies: pd.DataFrame | None = None
        self.sp500_index: pd.DataFrame | None = None

        self.adjusted_prices: pd.DataFrame | None = None
        self.close_prices: pd.DataFrame | None = None
        self.open_prices: pd.DataFrame | None = None
        self.high_prices: pd.DataFrame | None = None
        self.low_prices: pd.DataFrame | None = None
        self.volume: pd.DataFrame | None = None

        self.simple_returns: pd.DataFrame | None = None
        self.log_returns: pd.DataFrame | None = None
        self.metadata: pd.DataFrame | None = None

    def load(self) -> "SP500Data":
        """
        Download, parse, clean, and transform the S&P 500 dataset.

        Returns
        -------
        SP500Data
            The loaded object, allowing chained calls.
        """
        self.dataset_path = self._download_dataset()
        csv_files = self._find_csv_files(self.dataset_path)
        self._validate_files(csv_files)

        stocks, companies, sp500_index = self._read_csv_files(csv_files)

        self.stocks = self._clean_stocks(stocks)
        self.companies = self._clean_companies(companies)
        self.sp500_index = self._clean_index(sp500_index)

        self._create_market_matrices()
        self._calculate_returns()
        self._create_metadata()

        return self

    def _download_dataset(self) -> Path:
        dataset_path = Path(
            kagglehub.dataset_download(self.DATASET_NAME)
        )
        return dataset_path

    @staticmethod
    def _find_csv_files(dataset_path: Path) -> dict[str, Path]:
        return {
            file.name: file
            for file in dataset_path.glob("*.csv")
        }

    def _validate_files(self, csv_files: dict[str, Path]) -> None:
        missing_files = self.REQUIRED_FILES - set(csv_files)

        if missing_files:
            raise FileNotFoundError(
                f"Missing expected files: {sorted(missing_files)}"
            )

    @staticmethod
    def _read_csv_files(
        csv_files: dict[str, Path],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        stocks = pd.read_csv(
            csv_files["sp500_stocks.csv"],
            parse_dates=["Date"],
            low_memory=False,
        )

        companies = pd.read_csv(
            csv_files["sp500_companies.csv"],
            low_memory=False,
        )

        sp500_index = pd.read_csv(
            csv_files["sp500_index.csv"],
            parse_dates=["Date"],
            low_memory=False,
        )

        return stocks, companies, sp500_index

    def _clean_stocks(
        self,
        stocks: pd.DataFrame,
    ) -> pd.DataFrame:

        stocks = stocks.copy()
        stocks.columns = stocks.columns.str.strip()

        required_columns = {"Date", "Symbol"}
        missing_columns = required_columns - set(stocks.columns)

        if missing_columns:
            raise KeyError(
                f"Stock data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        stocks = (
            stocks
            .drop_duplicates(subset=["Date", "Symbol"])
            .sort_values(["Date", "Symbol"])
            .reset_index(drop=True)
        )

        for column in self.STOCK_NUMERIC_COLUMNS:
            if column in stocks.columns:
                stocks[column] = pd.to_numeric(
                    stocks[column],
                    errors="coerce",
                )

        return stocks

    @staticmethod
    def _clean_companies(
        companies: pd.DataFrame,
    ) -> pd.DataFrame:

        companies = companies.copy()
        companies.columns = companies.columns.str.strip()

        if "Symbol" not in companies.columns:
            raise KeyError(
                "Company data does not contain a 'Symbol' column."
            )

        return (
            companies
            .drop_duplicates(subset=["Symbol"])
            .sort_values("Symbol")
            .reset_index(drop=True)
        )

    @staticmethod
    def _clean_index(
        sp500_index: pd.DataFrame,
    ) -> pd.DataFrame:

        sp500_index = sp500_index.copy()
        sp500_index.columns = sp500_index.columns.str.strip()

        if "Date" not in sp500_index.columns:
            raise KeyError(
                "S&P 500 index data does not contain a 'Date' column."
            )

        return (
            sp500_index
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .set_index("Date")
        )

    def _create_market_matrices(self) -> None:
        stocks = self._require_dataframe(self.stocks, "stocks")

        self.adjusted_prices = self._make_matrix(
            stocks,
            "Adj Close",
        ).where(lambda values: values > 0)

        self.close_prices = self._make_matrix(
            stocks,
            "Close",
        ).where(lambda values: values > 0)

        self.open_prices = self._make_matrix(stocks, "Open")
        self.high_prices = self._make_matrix(stocks, "High")
        self.low_prices = self._make_matrix(stocks, "Low")
        self.volume = self._make_matrix(stocks, "Volume")

    def _calculate_returns(self) -> None:
        adjusted_prices = self._require_dataframe(
            self.adjusted_prices,
            "adjusted_prices",
        )

        self.simple_returns = (
            adjusted_prices
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )

        self.log_returns = (
            np.log(adjusted_prices / adjusted_prices.shift(1))
            .replace([np.inf, -np.inf], np.nan)
        )

    def _create_metadata(self) -> None:
        companies = self._require_dataframe(
            self.companies,
            "companies",
        )

        adjusted_prices = self._require_dataframe(
            self.adjusted_prices,
            "adjusted_prices",
        )

        metadata = companies.set_index("Symbol").sort_index()

        available_symbols = metadata.index.intersection(
            adjusted_prices.columns
        )

        self.metadata = metadata.reindex(available_symbols)

    @staticmethod
    def _make_matrix(
        stocks: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:

        if value_column not in stocks.columns:
            return pd.DataFrame(
                index=pd.Index([], name="Date")
            )

        return (
            stocks
            .pivot(
                index="Date",
                columns="Symbol",
                values=value_column,
            )
            .sort_index()
        )

    @staticmethod
    def _require_dataframe(
        dataframe: pd.DataFrame | None,
        name: str,
    ) -> pd.DataFrame:

        if dataframe is None:
            raise RuntimeError(
                f"{name!r} has not been initialized. "
                "Call load() first."
            )

        return dataframe

    def summary(self) -> pd.DataFrame:
        """
        Return the dimensions of all loaded DataFrames.
        """
        datasets = {
            "stocks": self.stocks,
            "companies": self.companies,
            "sp500_index": self.sp500_index,
            "adjusted_prices": self.adjusted_prices,
            "close_prices": self.close_prices,
            "open_prices": self.open_prices,
            "high_prices": self.high_prices,
            "low_prices": self.low_prices,
            "volume": self.volume,
            "simple_returns": self.simple_returns,
            "log_returns": self.log_returns,
            "metadata": self.metadata,
        }

        rows = []

        for name, dataframe in datasets.items():
            rows.append(
                {
                    "dataset": name,
                    "rows": (
                        len(dataframe)
                        if dataframe is not None
                        else None
                    ),
                    "columns": (
                        len(dataframe.columns)
                        if dataframe is not None
                        else None
                    ),
                }
            )

        return pd.DataFrame(rows).set_index("dataset")

    def get_clean_universe(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        minimum_coverage: float = 0.95,
        number_of_stocks: int | None = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        log_returns = self._require_dataframe(
            self.log_returns,
            "log_returns",
        )

        adjusted_prices = self._require_dataframe(
            self.adjusted_prices,
            "adjusted_prices",
        )

        metadata = self._require_dataframe(
            self.metadata,
            "metadata",
        )

        returns = log_returns.loc[start_date:end_date].copy()

        coverage = returns.notna().mean()
        symbols = coverage[
            coverage >= minimum_coverage
        ].index

        returns = returns.loc[:, symbols]
        prices = adjusted_prices.loc[
            returns.index,
            symbols,
        ]

        metadata = metadata.reindex(symbols)

        if (
            number_of_stocks is not None
            and len(symbols) > number_of_stocks
        ):
            if "Marketcap" not in metadata.columns:
                raise KeyError(
                    "Cannot select the largest companies because "
                    "'Marketcap' is missing from metadata."
                )

            selected_symbols = (
                metadata["Marketcap"]
                .dropna()
                .nlargest(number_of_stocks)
                .index
            )

            returns = returns.loc[:, selected_symbols]
            prices = prices.loc[:, selected_symbols]
            metadata = metadata.loc[selected_symbols]

        # Use dates for which every selected stock has a return.
        returns = returns.dropna()

        prices = prices.reindex(
            index=returns.index,
            columns=returns.columns,
        )

        metadata = metadata.reindex(returns.columns)

        return returns, prices, metadata