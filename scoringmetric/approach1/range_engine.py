# range_engine.py

import numpy as np

def derive_ranges(values):
    """
    Returns (ranges, rule) where:
    - ranges: None or list of boundaries/buckets
    - rule: 'discrete' | 'gap-based' | 'smooth'
    """

    values = np.asarray(values)
    unique_vals = np.unique(values)

    # 1. Low-cardinality / discrete feature
    if len(unique_vals) <= 6:
        return unique_vals.tolist(), "discrete"

    # 2. Density-gap based ranges
    sorted_vals = np.sort(values)
    diffs = np.diff(sorted_vals)
    gap_threshold = np.percentile(diffs, 97)
    split_points = sorted_vals[:-1][diffs > gap_threshold]

    if len(split_points) > 0:
        return split_points.tolist(), "gap-based"

    # 3. Smooth continuous → no ranges
    return None, "smooth"
