"""Simplified absolute-temperature heatwave rule (NOT the full IMD definition; never a regression target)."""
HEATWAVE_C = 45.0
SEVERE_C = 47.0
CATEGORY_NAMES = {0: 'No Heatwave', 1: 'Heatwave', 2: 'Severe Heatwave'}


def classify_heatwave(tmax_c):
    # Simplified project rule - NOT the full IMD definition (no departure-from-normal / coverage / duration).
    code = 2 if tmax_c >= SEVERE_C else 1 if tmax_c >= HEATWAVE_C else 0
    return code, CATEGORY_NAMES[code]

