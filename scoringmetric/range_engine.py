# range_engine.py

import numpy as np

def derive_ranges(values):
    """
    Returns:
    - None → no meaningful ranges
    - list → feature-specific boundaries or buckets
    """

    values = np.asarray(values)
    n = len(values)

    # 1. Low-cardinality / discrete feature
    unique_vals = np.unique(values)
    if len(unique_vals) <= 6:
        return unique_vals.tolist()

    # 2. Density-gap based ranges
    sorted_vals = np.sort(values)
    diffs = np.diff(sorted_vals)

    # Large gaps indicate natural breaks
    gap_threshold = np.percentile(diffs, 97)
    split_points = sorted_vals[:-1][diffs > gap_threshold]

    if len(split_points) > 0:
        return split_points.tolist()

    # 3. Smooth continuous → no ranges
    return None
