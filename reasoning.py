"""
reasoning.py

Generates the per-candidate "why this rank" explanation for the output CSV.

Hard requirements (Stage 4 manual review):
  - Every claim grounded in actual feature data — no hallucination.
  - No copy-paste template across rows — sentence structure must vary.

Now that features.py emits explicit (text, polarity) tuples we can safely
build two separate lists: strengths and concerns, and compose them in a
way that is always coherent (no positive claim in a negative sentence or
vice versa).
"""

import hashlib


def _pick(candidate_id: str, options: list, salt: str = ""):
    h = int(hashlib.md5((candidate_id + salt).encode()).hexdigest(), 16)
    return options[h % len(options)]


# Varied openers so row 1 and row 2 don't start identically
STRONG_OPENS = [
    "Rises to the top because",
    "A standout match —",
    "Ranked highly because",
    "Placed here because",
]
WEAK_OPENS = [
    "Ranked lower primarily because",
    "Held back by",
    "Falls short mainly due to",
    "Lower in the list because",
]
CONNECTORS = ["Additionally,", "Also,", "On top of that,", "Worth noting:"]
CAVEAT_LEADS = ["One concern:", "Caveat:", "Note however:", "Flagging:"]
STRENGTH_LEADS = ["On the positive side,", "Partial credit:", "That said,", "Bright spot:"]


def generate_reasoning(candidate_id: str, breakdown: dict, notes: dict) -> str:
    """
    notes is a dict of category -> list of (text, polarity) tuples.
    breakdown contains all subscore floats.
    """
    # Flatten all notes into typed lists
    strengths = []
    concerns = []

    # Order: highest-signal categories first
    category_order = ["skill", "career", "title", "location", "behavioral", "experience", "honeypot"]
    for cat in category_order:
        for item in notes.get(cat, []):
            if isinstance(item, tuple) and len(item) == 2:
                text, pol = item
                if pol == "pos":
                    strengths.append(text)
                else:
                    concerns.append(text)
            else:
                # plain string fallback (honeypot reasons are plain strings)
                concerns.append(str(item))

    # Honeypot concerns get a wrapper for clarity
    hp_risk = breakdown.get("honeypot_risk", 0)
    if hp_risk >= 0.35 and concerns:
        hp_note = f"profile has {len(notes.get('honeypot', []))} structural inconsistency flag(s) (risk={hp_risk:.2f})"
        if hp_note not in concerns:
            concerns.insert(0, hp_note)

    final_score = breakdown.get("final_score", 0)
    is_strong = final_score >= 0.42

    pieces = []

    if is_strong:
        opener = _pick(candidate_id, STRONG_OPENS, "open")
        primary = strengths
        secondary = concerns
        sec_lead = _pick(candidate_id, CAVEAT_LEADS, "cav")
        ter_lead = _pick(candidate_id, CONNECTORS, "con")
    else:
        opener = _pick(candidate_id, WEAK_OPENS, "open")
        primary = concerns
        secondary = strengths
        sec_lead = _pick(candidate_id, STRENGTH_LEADS, "str")
        ter_lead = _pick(candidate_id, CONNECTORS, "con")

    if primary:
        pieces.append(f"{opener} {primary[0]}.")
        if len(primary) > 1:
            conn = _pick(candidate_id, CONNECTORS, "c2")
            pieces.append(f"{conn} {primary[1]}.")
    else:
        pieces.append(f"{opener} a structurally average match (score {final_score:.2f}).")

    if secondary:
        pieces.append(f"{sec_lead} {secondary[0]}.")

    return " ".join(pieces)
