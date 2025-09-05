from __future__ import annotations
import pandas as pd

def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    results: dict of { model_name: metrics_dict }
    """
    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE")
