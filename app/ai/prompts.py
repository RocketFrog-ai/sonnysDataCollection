from .signals import BANDED_FEATURES

def _band_section():
    lines = [
        "Features with defined impact bands (use these ranges to set normalized 0-1 and strength):",
        "",
    ]
    for key, label, bands in BANDED_FEATURES:
        lines.append(f"- {key} ({label}): {bands}.")
    lines.append("")
    lines.append('For any other feature (distances, counts, precipitation, windspeed, rating, etc.): normalize the value to 0–1 based on typical ranges and whether higher or lower is better; set strength to "strong", "moderate", or "weak" based on how extreme the value is.')
    return "\n".join(lines)

def build_normalized_strength_prompt(feature_values):
    lines = [
        "You are an agent that scores site features. For each feature below, output a normalized score from 0 to 1 and a strength: strong, moderate, or weak.",
        _band_section(),
        "",
        "Current feature values:",
    ]
    for name, val in feature_values.items():
        if val is None:
            continue
        val_str = str(int(val)) if isinstance(val, float) and val == int(val) else (f"{val:.2f}" if isinstance(val, float) else str(val))
        lines.append(f"  - {name}: {val_str}")
    lines.extend([
        "",
        "Respond with ONLY a single JSON object. One key per feature name above. Each value must be: {\"normalized\": <number 0-1>, \"strength\": \"strong\"|\"moderate\"|\"weak\"}.",
        "Example: {\"sunny_days_per_year\": {\"normalized\": 0.75, \"strength\": \"moderate\"}, ...}",
        "Do not include any text before or after the JSON.",
    ])
    return "\n".join(lines)

def _feature_reference_and_values(feature_values):
    if not feature_values:
        return ""
    parts = ["Available features and impact ranges (Agentic AI gives impact score):"]
    for key, label, bands in BANDED_FEATURES:
        val = feature_values.get(key)
        if val is None:
            continue
        fmt = ".1f" if key == "sunny_days_per_year" else ".0f"
        parts.append(f"  - {key} ({label}): {val:{fmt}}")
        parts.append(f"    Impact: {bands}")
    if len(parts) == 1:
        return ""
    return "\n".join(parts)

def build_rationale_prompt(pros_text, cons_text, feature_values=None):
    ref_block = _feature_reference_and_values(feature_values or {})
    ref_section = f"\n{ref_block}\n" if ref_block else ""
    return f"""Analyze this car wash site location based on the following features and correlations.
{ref_section}
Positive Factors:
{pros_text}

Negative Factors:
{cons_text}

Provide a concise rationale (2-3 sentences) explaining the overall assessment of this site location for a car wash business. Focus on the most impactful factors."""

def build_pros_cons_prompt(pros_data, cons_data):
    return f"""Based on these car wash site location features, generate specific pros and cons.

Positive Features:
{pros_data}

Negative Features:
{cons_data}

Generate 3-5 specific pros and 3-5 specific cons. Be specific about why each factor matters for a car wash business.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "pros": ["specific pro 1", "specific pro 2", "specific pro 3"],
  "cons": ["specific con 1", "specific con 2", "specific con 3"]
}}

Do not include any text before or after the JSON. Only return the JSON object."""
