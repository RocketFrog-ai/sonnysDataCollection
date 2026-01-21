def build_rationale_prompt(pros_text: str, cons_text: str) -> str:
    return f"""Analyze this car wash site location based on the following features and correlations.

Positive Factors:
{pros_text}

Negative Factors:
{cons_text}

Provide a concise rationale (2-3 sentences) explaining the overall assessment of this site location for a car wash business. Focus on the most impactful factors."""


def build_pros_cons_prompt(pros_data: str, cons_data: str) -> str:
    return f"""Based on these car wash site location features, generate specific pros and cons.

Positive Features:
{pros_data}

Negative Features:
{cons_data}

Generate 3-5 specific pros and 3-5 specific cons as bullet points. Be specific about why each factor matters for a car wash business. Format as:
PROS:
- [specific pro 1]
- [specific pro 2]

CONS:
- [specific con 1]
- [specific con 2]"""
