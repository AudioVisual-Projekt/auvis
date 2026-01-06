import pandas as pd
import os
from typing import Optional


def load_overview_data(project_root: str, dataset: str = "dev") -> pd.DataFrame:
    """
    Lädt die 'overview_{dataset}.csv' und bereitet sie für Analysen vor.
    """
    csv_path = os.path.join(project_root, "data-bin", "_output", "av_asd", dataset, f"overview_{dataset}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Optional: Filtere Spalten, um das DataFrame lesbarer zu machen
    # Wir behalten Metriken, Parameter und alle Session-Spalten
    return df


def get_session_columns(df: pd.DataFrame, suffix: str = "_eva") -> list:
    """
    Gibt eine Liste aller Spalten zurück, die Sessions betreffen.
    z.B. suffix="_eva" -> ['s138_eva', 's140_eva', ...]
    """
    return [col for col in df.columns if col.endswith(suffix)]


def filter_top_experiments(df: pd.DataFrame, metric: str = "AVG_SPEAKER_F1", percentile: float = 0.2) -> pd.DataFrame:
    """
    Gibt nur die besten X% der Experimente zurück (für Hard Sample Mining).
    """
    # Berechne den Cutoff-Wert (z.B. das 80. Perzentil)
    cutoff = df[metric].quantile(1.0 - percentile)
    return df[df[metric] >= cutoff].copy()
