"""
features.py

Structured (non-semantic) scoring components. Each function returns a value
in [0, 1] plus a list of (text, polarity) note tuples, polarity in
{"pos", "neg"}. Tagging polarity explicitly at the source — instead of having
reasoning.py infer it from whether the parent subscore crossed some threshold —
avoids contradictions like "strong on core skills... weak/no evidence of core
skills" that appeared in early testing.

Design principle: every rule traces back to a specific clause in
job_description.md (see jd_profile.py for the citations).
"""

from datetime import date, datetime
import jd_profile as jp


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def all_text(candidate: dict) -> str:
    """Lowercased blob of every free-text field, for keyword scanning."""
    parts = []
    profile = candidate.get("profile", {}) or {}
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    parts.append(profile.get("current_industry", ""))
    for job in candidate.get("career_history", []) or []:
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))
        parts.append(job.get("company", ""))
    for sk in candidate.get("skills", []) or []:
        parts.append(sk.get("name", ""))
    return " | ".join(p for p in parts if p).lower()


# ---------------------------------------------------------------------------
# Skill match (trust-weighted, not just keyword presence)
# ---------------------------------------------------------------------------
def skill_match_trust(candidate: dict):
    skills = candidate.get("skills", []) or []
    skill_index = {s.get("name", "").lower(): s for s in skills}
    text = all_text(candidate)

    PROF_WEIGHT = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    def trust_weighted_hit(keyword: str) -> float:
        for name, sk in skill_index.items():
            if keyword in name or name in keyword:
                prof = PROF_WEIGHT.get(sk.get("proficiency", "beginner"), 0.3)
                dur = min((sk.get("duration_months") or 0) / 24.0, 1.0)
                endorsed = min((sk.get("endorsements") or 0) / 20.0, 1.0)
                trust = 0.5 * prof + 0.3 * dur + 0.2 * endorsed
                return max(0.3, trust)
        if keyword in text:
            return 0.2
        return 0.0

    core_hits = [trust_weighted_hit(k) for k in jp.CORE_REQUIRED_SKILLS]
    nice_hits = [trust_weighted_hit(k) for k in jp.NICE_TO_HAVE_SKILLS]

    # Coverage across distinct skill areas matters more than raw average
    core_coverage = sum(1 for v in core_hits if v >= 0.3) / len(jp.CORE_REQUIRED_SKILLS)
    hits_above_zero = [v for v in core_hits if v > 0]
    core_quality = sum(hits_above_zero) / len(hits_above_zero) if hits_above_zero else 0.0
    nice_coverage = sum(1 for v in nice_hits if v >= 0.3) / len(jp.NICE_TO_HAVE_SKILLS)

    score = min(1.0, 0.65 * core_coverage * 3.0 + 0.20 * core_quality + 0.15 * nice_coverage * 3.0)
    score = max(0.0, min(1.0, score))

    notes = []
    top_core = sorted(zip(jp.CORE_REQUIRED_SKILLS, core_hits), key=lambda x: -x[1])
    strong_core = [k for k, v in top_core if v >= 0.5][:4]

    if strong_core:
        notes.append((f"credible evidence of core skills: {', '.join(strong_core)}", "pos"))
    elif not any(v > 0 for v in core_hits):
        notes.append(("no evidence of any core retrieval/ranking/vector-search skill", "neg"))
    else:
        notes.append(("only weak/text-mention evidence of core retrieval skills — nothing endorsed or long-held", "neg"))

    strong_nice = [k for k, v in zip(jp.NICE_TO_HAVE_SKILLS, nice_hits) if v >= 0.5]
    if strong_nice:
        notes.append((f"also brings nice-to-have skills: {', '.join(strong_nice[:3])}", "pos"))

    return score, notes


# ---------------------------------------------------------------------------
# Title / seniority fit
# ---------------------------------------------------------------------------
def title_seniority_fit(candidate: dict):
    profile = candidate.get("profile", {}) or {}
    title = (profile.get("current_title") or "").lower()

    ai_ml_title_terms = ["ai engineer", "ml engineer", "machine learning",
                          "applied scientist", "search", "ranking", "retrieval",
                          "nlp engineer", "data scientist", "ai/ml", "research engineer"]
    is_ai_title = any(t in title for t in ai_ml_title_terms)

    clearly_unrelated = ["marketing", "sales", "hr ", "human resources",
                          "recruiter", "accountant", "designer (visual)",
                          "content writer", "graphic designer", "office",
                          "operations manager", "business development"]
    is_unrelated_title = any(t in title for t in clearly_unrelated)

    notes = []
    if is_unrelated_title:
        notes.append((f"current title '{profile.get('current_title')}' is unrelated to AI engineering", "neg"))
        return 0.05, notes

    if is_ai_title:
        score = 0.85
        notes.append((f"title '{profile.get('current_title')}' directly aligns with the AI Engineer mandate", "pos"))
    else:
        engineering_terms = ["engineer", "developer", "architect", "scientist"]
        if any(t in title for t in engineering_terms):
            score = 0.45
            notes.append((f"title '{profile.get('current_title')}' is engineering-adjacent but not AI-labeled", "neg"))
        else:
            score = 0.15
            notes.append((f"title '{profile.get('current_title')}' has no clear engineering signal", "neg"))

    if any(t in title for t in jp.NON_CODING_TITLE_KEYWORDS):
        career = candidate.get("career_history", []) or []
        current = next((j for j in career if j.get("is_current")), None)
        if current:
            sd = _parse_date(current.get("start_date"))
            if sd and (date.today() - sd).days > 18 * 30:
                score *= 0.55
                notes.append(("in a non-coding leadership title for 18+ months; JD role is hands-on", "neg"))

    return max(0.0, min(1.0, score)), notes


# ---------------------------------------------------------------------------
# Career quality
# ---------------------------------------------------------------------------
def career_quality(candidate: dict):
    career = candidate.get("career_history", []) or []
    text = all_text(candidate)
    notes = []
    score = 0.55

    research_signal = any(r in text for r in jp.RESEARCH_ONLY_SIGNALS)
    production_signal = any(p in text for p in jp.PRODUCTION_SIGNALS)
    if research_signal and not production_signal:
        notes.append(("research-only background, no evidence of production deployment (JD hard disqualifier)", "neg"))
        return 0.05, notes

    companies = [j.get("company", "").lower() for j in career]
    industries = [j.get("industry", "").lower() for j in career]
    all_consulting = bool(companies) and all(
        any(cf in c for cf in jp.CONSULTING_FIRMS) or "it services" in ind
        for c, ind in zip(companies, industries)
    )
    if all_consulting:
        score -= 0.35
        notes.append(("entire career at consulting/IT-services firms only (JD penalizes this)", "neg"))

    nlp_hits = sum(1 for s in jp.NLP_IR_SIGNALS if s in text)
    domain_hits = sum(1 for s in jp.NON_NLP_DOMAIN_SIGNALS if s in text)
    if domain_hits >= 2 and nlp_hits == 0:
        score -= 0.3
        notes.append(("CV/speech/robotics background with no NLP/IR exposure (JD flags re-learning risk)", "neg"))
    elif nlp_hits >= 2:
        score += 0.2
        notes.append(("clear NLP/retrieval/ranking exposure across career history", "pos"))

    ladder_roles = [
        j for j in career
        if any(k in (j.get("title") or "").lower() for k in jp.TITLE_LADDER_KEYWORDS)
    ]
    short_ladder_hops = sum(1 for j in ladder_roles if (j.get("duration_months") or 99) < 18)
    if len(career) >= 3 and short_ladder_hops >= 2:
        score -= 0.2
        notes.append((f"{short_ladder_hops} senior-ladder roles held under 18 months (title-chasing pattern)", "neg"))

    if any(k in text for k in ["scale", "millions of", "production", "real-time", "shipped"]):
        score += 0.1
        notes.append(("language suggesting real production-scale ownership", "pos"))

    return max(0.0, min(1.0, score)), notes


# ---------------------------------------------------------------------------
# Experience-years fit
# ---------------------------------------------------------------------------
def experience_fit(candidate: dict):
    yoe = (candidate.get("profile", {}) or {}).get("years_of_experience") or 0
    if jp.EXPERIENCE_IDEAL_MIN <= yoe <= jp.EXPERIENCE_IDEAL_MAX:
        return 1.0, [(f"{yoe} yrs experience is squarely in the JD's 5-9 yr band", "pos")]
    if jp.EXPERIENCE_SOFT_MIN <= yoe < jp.EXPERIENCE_IDEAL_MIN:
        score = 0.5 + 0.5 * (yoe - jp.EXPERIENCE_SOFT_MIN) / (jp.EXPERIENCE_IDEAL_MIN - jp.EXPERIENCE_SOFT_MIN)
        return score, [(f"{yoe} yrs is a bit below the ideal band (JD says flexible if signals are strong)", "neg")]
    if jp.EXPERIENCE_IDEAL_MAX < yoe <= jp.EXPERIENCE_SOFT_MAX:
        score = 1.0 - 0.4 * (yoe - jp.EXPERIENCE_IDEAL_MAX) / (jp.EXPERIENCE_SOFT_MAX - jp.EXPERIENCE_IDEAL_MAX)
        return score, [(f"{yoe} yrs is slightly above the ideal band", "neg")]
    if yoe < jp.EXPERIENCE_SOFT_MIN:
        return 0.15, [(f"only {yoe} yrs experience, well below the JD band", "neg")]
    return 0.35, [(f"{yoe} yrs, well above the ideal band (overqualified / architect-drift risk)", "neg")]


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------
def location_fit(candidate: dict):
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = signals.get("willing_to_relocate", False)

    if country and country != "india":
        if willing:
            return 0.25, [(f"based in {profile.get('country')}; willing to relocate but JD offers no visa sponsorship", "neg")]
        return 0.05, [(f"based in {profile.get('country')}, not willing to relocate; JD does not sponsor visas", "neg")]

    if any(c in location for c in jp.PRIMARY_LOCATIONS):
        return 1.0, [(f"already based in {profile.get('location')}, a JD-preferred location", "pos")]
    if any(c in location for c in jp.WELCOME_LOCATIONS):
        return 0.85, [(f"based in {profile.get('location')}, explicitly welcomed by the JD", "pos")]
    if any(c in location for c in jp.TIER1_INDIAN_CITIES):
        if willing:
            return 0.65, [(f"Tier-1 city ({profile.get('location')}) and open to relocating", "pos")]
        return 0.4, [(f"Tier-1 city ({profile.get('location')}) but not confirmed willing to relocate", "neg")]
    if willing:
        return 0.35, [(f"based in {profile.get('location')} (non-Tier-1) but willing to relocate", "neg")]
    return 0.15, [(f"based in {profile.get('location')}, not a JD-preferred location and not willing to relocate", "neg")]


# ---------------------------------------------------------------------------
# Behavioral reliability — multiplicative, not additive
# ---------------------------------------------------------------------------
def behavioral_multiplier(candidate: dict, today: date = None):
    today = today or date.today()
    s = candidate.get("redrob_signals", {}) or {}
    notes = []
    mult = 1.0

    last_active = _parse_date(s.get("last_active_date"))
    if last_active:
        days_inactive = (today - last_active).days
        if days_inactive > 180:
            mult *= 0.55
            notes.append((f"inactive for {days_inactive}d (>6 months) — likely unreachable", "neg"))
        elif days_inactive > 90:
            mult *= 0.8
            notes.append((f"inactive for {days_inactive}d", "neg"))
        elif days_inactive <= 14:
            mult *= 1.05
            notes.append(("active on platform within the last 2 weeks", "pos"))

    if s.get("open_to_work_flag") is False:
        mult *= 0.6
        notes.append(("not currently flagged open-to-work", "neg"))

    rr = s.get("recruiter_response_rate")
    if rr is not None:
        mult *= (0.7 + 0.3 * rr)
        if rr < 0.15:
            notes.append((f"very low recruiter response rate ({rr:.0%})", "neg"))
        elif rr > 0.7:
            notes.append((f"high recruiter responsiveness ({rr:.0%})", "pos"))

    icr = s.get("interview_completion_rate")
    if icr is not None and icr < 0.5:
        mult *= 0.85
        notes.append((f"low interview-completion rate ({icr:.0%})", "neg"))

    notice = s.get("notice_period_days")
    if notice is not None:
        if notice <= jp.NOTICE_PERIOD_IDEAL_DAYS:
            mult *= 1.05
            notes.append((f"short notice period ({notice}d)", "pos"))
        elif notice > 60:
            mult *= 0.85
            notes.append((f"long notice period ({notice}d)", "neg"))

    if not s.get("verified_email", True) and not s.get("verified_phone", True):
        mult *= 0.9
        notes.append(("neither email nor phone verified", "neg"))

    return max(0.2, min(1.15, mult)), notes
