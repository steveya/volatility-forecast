"""Feature selection utilities for STES and XGBSTES variants."""

import pandas as pd


def infer_feature_group(col: str) -> str:
    """Infer feature group from column name."""
    lc = col.lower()
    if "abs" in lc:
        return "abs"
    if "squared" in lc or "sq" in lc:
        return "sq"
    if "log" in lc or "ret" in lc:
        return "raw"
    return "raw"  # default


def select_variant_columns(
    X: pd.DataFrame, variant: str, variant_groups: dict[str, set[str]] | None = None
) -> list[str]:
    """Select columns for a variant; include const when present.

    Args:
        X: Feature DataFrame
        variant: Name of the variant
        variant_groups: Optional mapping of variant -> set of feature groups needed.
                       If None, uses default mapping or infers from variant name.

    Returns:
        List of column names to use for this variant
    """
    # Default mapping for known variants
    default_groups = {
        "ES": set(),
        "STES_AE": {"abs"},
        "STES_SE": {"sq"},
        "STES_EAE": {"raw", "abs"},
        "STES_ESE": {"raw", "sq"},
        "STES_AESE": {"abs", "sq"},
        "STES_EAESE": {"raw", "abs", "sq"},
        # Option B: unified naming for all XGBSTES variants
        "XGBSTES_BASE": {"raw", "abs", "sq"},
        "XGBSTES_BASE_MONO": {"raw", "abs", "sq"},
        "XGBSTES_BASE_HUBER": {"raw", "abs", "sq"},
        "XGBSTES_BASE_RESID": {"raw", "abs", "sq"},
        "XGBSTES_BASE_MONO_HUBER": {"raw", "abs", "sq"},
        "XGBSTES_BASE_MONO_RESID": {"raw", "abs", "sq"},
        "XGBSTES_BASE_HUBER_RESID": {"raw", "abs", "sq"},
        "XGBSTES_BASE_MONO_HUBER_RESID": {"raw", "abs", "sq"},
    }

    # Use provided mapping or default
    if variant_groups is not None:
        groups_map = variant_groups
    else:
        groups_map = default_groups

    # Get groups for this variant, default to all features if unknown
    if variant in groups_map:
        groups_needed = groups_map[variant]
    else:
        # Unknown variant: infer or default to all features for tree models
        if variant.startswith("XGBSTES"):
            groups_needed = {"raw", "abs", "sq"}
        elif variant.startswith("XGB"):
            # Backward-compatible catch-all for any remaining XGB* names.
            groups_needed = {"raw", "abs", "sq"}
        else:
            # Try to infer from STES naming convention
            groups_needed = _infer_groups_from_stes_name(variant)

    cols = []
    for c in X.columns:
        if c == "const":
            continue
        g = infer_feature_group(c)
        if g in groups_needed:
            cols.append(c)

    if variant == "ES" or len(groups_needed) == 0:
        return ["const"] if "const" in X.columns else []

    if "const" in X.columns:
        cols = ["const"] + cols
    return cols


def _infer_groups_from_stes_name(variant: str) -> set[str]:
    """Infer feature groups from STES variant naming convention.

    Convention: E=raw, A=abs, S=squared
    Example: STES_EAE uses raw and abs features
    """
    groups = set()
    variant_upper = variant.upper()

    if "E" in variant_upper and "STES" in variant_upper:
        # Check for E in the suffix after STES_
        parts = variant_upper.split("_")
        if len(parts) > 1:
            suffix = parts[-1]
            if "E" in suffix and suffix not in ["SE", "ESE", "AESE", "EAESE"]:
                groups.add("raw")

    if "A" in variant_upper:
        groups.add("abs")

    if "S" in variant_upper and "STES" in variant_upper:
        parts = variant_upper.split("_")
        if len(parts) > 1 and "S" in parts[-1]:
            groups.add("sq")

    # If nothing inferred, default to all
    if not groups:
        groups = {"raw", "abs", "sq"}

    return groups
