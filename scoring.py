"""
scoring.py

Combines semantic fit + structured features into one composite score.

Composite = weighted_sum(semantic_fit, skill_match_trust, title_seniority_fit,
                          career_quality, experience_fit, location_fit)
            * behavioral_multiplier
            * (1 - honeypot_risk)

Why multiplicative for the last two terms instead of folding them into the
weighted sum: a candidate who is a great structural match but is an
honeypot (fabricated profile) or behaviorally unreachable (inactive 8
months, never responds) isn't "70% as good" — they're not a usable
recommendation. Multiplying lets these factors suppress an otherwise high
score without us having to hand-pick yet another additive weight.
"""

import jd_profile as jp
import features as ft
import honeypot as hp


def score_structural(candidate: dict):
    """Everything except semantic_fit (which needs the corpus-wide TF-IDF
    matrix and is computed separately/vectorized in rank.py for speed).
    Returns the partial weighted base (5 of 6 terms), behavioral multiplier,
    honeypot risk, and the notes dict needed for reasoning generation.
    """
    skill_score, skill_notes = ft.skill_match_trust(candidate)
    title_score, title_notes = ft.title_seniority_fit(candidate)
    career_score, career_notes = ft.career_quality(candidate)
    exp_score, exp_notes = ft.experience_fit(candidate)
    loc_score, loc_notes = ft.location_fit(candidate)
    behav_mult, behav_notes = ft.behavioral_multiplier(candidate)
    risk, risk_reasons = hp.detect_honeypot(candidate)

    base_partial = (
        jp.WEIGHTS["skill_match_trust"] * skill_score
        + jp.WEIGHTS["title_seniority_fit"] * title_score
        + jp.WEIGHTS["career_quality"] * career_score
        + jp.WEIGHTS["experience_fit"] * exp_score
        + jp.WEIGHTS["location_fit"] * loc_score
    )

    partial_breakdown = {
        "skill_match_trust": round(skill_score, 4),
        "title_seniority_fit": round(title_score, 4),
        "career_quality": round(career_score, 4),
        "experience_fit": round(exp_score, 4),
        "location_fit": round(loc_score, 4),
        "behavioral_multiplier": round(behav_mult, 4),
        "honeypot_risk": round(risk, 4),
    }
    notes = {
        "skill": skill_notes,
        "title": title_notes,
        "career": career_notes,
        "experience": exp_notes,
        "location": loc_notes,
        "behavioral": behav_notes,
        "honeypot": risk_reasons,
    }
    return base_partial, behav_mult, risk, partial_breakdown, notes


def finalize_score(base_partial: float, semantic_sim: float, behav_mult: float, risk: float):
    base = base_partial + jp.WEIGHTS["semantic_fit"] * semantic_sim
    final = base * behav_mult * (1.0 - risk)
    return max(0.0, min(1.0, final)), round(base, 4)


def score_candidate(candidate: dict, semantic_sim: float):
    """Convenience single-shot version (used in tests / small samples)."""
    base_partial, behav_mult, risk, partial_breakdown, notes = score_structural(candidate)
    final, base = finalize_score(base_partial, semantic_sim, behav_mult, risk)
    breakdown = dict(partial_breakdown)
    breakdown["semantic_fit"] = round(semantic_sim, 4)
    breakdown["base_weighted_score"] = base
    breakdown["final_score"] = round(final, 4)
    return final, breakdown, notes
