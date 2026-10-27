"""Bonus: VAR/VECM analysis of the three supplied datasets.

Run from this directory with:
    python bonus_vecm_analysis.py

The script writes ``energy_vecm_rolling_forecast.png`` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASETS = {
    "energy": DATA_DIR / "energy.csv",
    "Precious_Metals": DATA_DIR / "Precious_Metals.csv",
    "Technology": DATA_DIR / "Technology.csv",
}


def adf_pvalue(series: pd.Series) -> float:
    """Return the ADF p-value for a series, using AIC lag selection."""
    return float(adfuller(series.dropna(), autolag="AIC")[1])


def johansen_rank(levels: pd.DataFrame) -> tuple[int, object]:
    """Determine rank using the first non-rejected 5% trace-test hypothesis."""
    result = coint_johansen(levels, det_order=0, k_ar_diff=1)
    for rank, (trace_stat, critical_value) in enumerate(
        zip(result.lr1, result.cvt[:, 1])
    ):
        if trace_stat <= critical_value:
            return rank, result
    return levels.shape[1], result


def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - predicted.to_numpy())))


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    errors = actual.to_numpy() - predicted.to_numpy()
    return float(np.sqrt(np.mean(errors**2)))


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    levels = frame.select_dtypes(include="number").dropna()
    if levels.shape[1] < 2:
        raise ValueError(f"{path.name} must contain at least two numeric series.")
    return levels


def main() -> None:
    results = {}
    summary_rows = []

    for name, path in DATASETS.items():
        levels = load_dataset(path)
        rank, johansen = johansen_rank(levels)
        level_pvalues = {column: adf_pvalue(levels[column]) for column in levels}
        difference_pvalues = {
            column: adf_pvalue(levels[column].diff()) for column in levels
        }
        results[name] = {
            "levels": levels,
            "rank": rank,
            "johansen": johansen,
            "level_pvalues": level_pvalues,
            "difference_pvalues": difference_pvalues,
        }
        summary_rows.append(
            {
                "dataset": name,
                "observations": len(levels),
                "variables": levels.shape[1],
                "cointegration rank r": rank,
                "level ADF p-values": ", ".join(
                    f"{column}: {value:.4f}"
                    for column, value in level_pvalues.items()
                ),
                "difference ADF p-values": ", ".join(
                    f"{column}: {value:.4f}"
                    for column, value in difference_pvalues.items()
                ),
                "recommended model": "VECM" if rank > 0 else "VAR in first differences",
            }
        )

        trace_table = pd.DataFrame(
            {
                "H0": [f"r <= {i}" for i in range(len(johansen.lr1))],
                "trace statistic": johansen.lr1,
                "5% critical value": johansen.cvt[:, 1],
                "reject at 5%": johansen.lr1 > johansen.cvt[:, 1],
            }
        )
        print(f"\n{name}: Johansen trace test")
        print(trace_table.round(4).to_string(index=False))

    summary = pd.DataFrame(summary_rows).set_index("dataset")
    print("\nClassification summary")
    print(summary.to_string())
    print(
        "\nInterpretation: r is the number of independent stationary linear "
        "combinations of the non-stationary level series."
    )
    print(
        "A positive rank means that a VECM can represent both short-run "
        "changes and long-run equilibrium correction."
    )

    vecm_candidates = [name for name, item in results.items() if item["rank"] > 0]
    if len(vecm_candidates) != 1:
        raise ValueError(
            f"Expected exactly one VECM candidate, found: {vecm_candidates}"
        )

    vecm_name = vecm_candidates[0]
    levels = results[vecm_name]["levels"]
    rank = results[vecm_name]["rank"]

    # The final three calendar months are held out; every forecast is one step ahead.
    test_start = levels.index.max() - pd.DateOffset(months=3)
    train = levels.loc[levels.index < test_start]
    test = levels.loc[levels.index >= test_start]
    if len(train) < 30 or test.empty:
        raise ValueError("The selected dataset is too short for the requested split.")

    lag_order = select_order(train, maxlags=10, deterministic="ci")
    k_ar_diff = lag_order.aic
    if k_ar_diff is None or not np.isfinite(k_ar_diff):
        raise ValueError("AIC did not produce a valid VECM lag order.")
    k_ar_diff = int(k_ar_diff)

    fitted_model = VECM(
        train,
        k_ar_diff=k_ar_diff,
        coint_rank=rank,
        deterministic="ci",
    ).fit()

    predictions = []
    for date in test.index:
        history = levels.loc[levels.index < date]
        rolling_fit = VECM(
            history,
            k_ar_diff=k_ar_diff,
            coint_rank=rank,
            deterministic="ci",
        ).fit()
        predictions.append(rolling_fit.predict(steps=1)[0])

    predicted = pd.DataFrame(predictions, index=test.index, columns=levels.columns)
    metrics = pd.DataFrame(
        {
            "MAE": {
                column: mean_absolute_error(test[column], predicted[column])
                for column in levels.columns
            },
            "RMSE": {
                column: root_mean_squared_error(test[column], predicted[column])
                for column in levels.columns
            },
        }
    )

    print(f"\nVECM dataset: {vecm_name}")
    print(f"Cointegration rank: r = {rank}")
    print(f"Training period: {train.index.min().date()} to {train.index.max().date()}")
    print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")
    print(f"Selected k_ar_diff by AIC: {k_ar_diff}")
    print("\nRolling-origin one-step forecast metrics")
    print(metrics.round(4).to_string())

    fig, axes = plt.subplots(
        len(levels.columns),
        1,
        figsize=(12, 3 * len(levels.columns)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, levels.columns):
        axis.plot(test.index, test[column], label="Skutočná hodnota")
        axis.plot(test.index, predicted[column], label="Predikcia")
        axis.set_ylabel(column)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Dátum")
    fig.suptitle(f"Rolling-origin VECM predikcia: {vecm_name}")
    fig.tight_layout()
    output_path = BASE_DIR / "energy_vecm_rolling_forecast.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"\nGraph saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
